import os
import threading
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Callable, Optional

import numpy as np
from PySide6 import QtCore

from src.pipeline import SpeechToEnglishPipeline, Translator
from src.pipeline_impl import JaToEnTranslator, EnToJaTranslator
from src.lfm2_transcriber import LFM2AudioTranscriber
from src.whisper_transcriber import WhisperOVTranscriber
from src.vad import SileroVAD

_DEBUG_AUDIO = os.getenv("AUDIO_DEBUG", "0") not in ("", "0", "false", "False")
_BYPASS_VAD = os.getenv("AUDIO_BYPASS_VAD", "0") not in ("", "0", "false", "False")


def _debug(msg: str):
    if _DEBUG_AUDIO:
        print(f"[audio] {msg}", flush=True)


@dataclass(frozen=True)
class RunConfig:
    use_loopback: bool
    source_lang: str
    chunk_seconds: float
    engine: str
    lfm2_repo: str
    enable_short: bool
    enable_translation: bool


class PipelineFactory:
    def create(self, cfg: RunConfig) -> SpeechToEnglishPipeline:
        if cfg.engine == "lfm2":
            transcriber = LFM2AudioTranscriber(repo_id=cfg.lfm2_repo or None)
        else:
            transcriber = WhisperOVTranscriber(
                language=cfg.source_lang,
                task="transcribe",
                device="GPU",
            )
        translator = JaToEnTranslator() if cfg.source_lang == "ja" else EnToJaTranslator()
        return SpeechToEnglishPipeline(transcriber, translator)

    def create_translator(self, source_lang: str) -> Translator:
        return JaToEnTranslator() if source_lang == "ja" else EnToJaTranslator()


class AudioProducer:
    def __init__(self, record_fn: Callable[..., tuple[np.ndarray, int]]):
        self._record_fn = record_fn

    def run(
        self,
        stop_event: threading.Event,
        out_short: Optional[Queue],
        target_sr: int,
        enable_short: Callable[[], bool],
        vad: Optional[SileroVAD],
        get_chunk_seconds: Callable[[], float],
        use_loopback: bool,
    ):
        while not stop_event.is_set():
            audio, sr = self._record_fn(
                duration_s=get_chunk_seconds(),
                target_sr=target_sr,
                loopback=use_loopback,
            )
            if audio.size == 0:
                _debug("producer got empty audio chunk")
                continue
            if not _BYPASS_VAD and vad is not None and not vad.has_speech(audio, sr):
                _debug("producer dropped chunk: VAD no speech")
                continue
            if out_short is not None and enable_short():
                out_short.put((audio, sr))


class ShortProcessor:
    def __init__(
        self,
        get_pipeline: Callable[[], Optional[SpeechToEnglishPipeline]],
        get_source_lang: Callable[[], str],
        get_translator: Callable[[str], Translator],
    ):
        self._get_pipeline = get_pipeline
        self._get_source_lang = get_source_lang
        self._get_translator = get_translator

    def run(
        self,
        stop_event: threading.Event,
        in_q: Queue,
        emit: Callable[[str, str, Optional[str], str], None],
        enable_translation: Callable[[], bool],
    ):
        while not stop_event.is_set() or not in_q.empty():
            try:
                audio, sr = in_q.get(timeout=0.5)
            except Empty:
                continue
            if audio.size == 0:
                in_q.task_done()
                continue
            pipeline = self._get_pipeline()
            if pipeline is None:
                in_q.task_done()
                continue
            source_lang = self._get_source_lang()
            source_text = pipeline.recognizer.transcribe_array(audio, sr)
            translator = self._get_translator(
                source_lang
            )
            translated = (
                translator.translate(source_text)
                if source_text and enable_translation()
                else ""
            )
            emit(source_text, translated, None, source_lang)
            in_q.task_done()


class SpeechEngine(QtCore.QObject):
    text_ready_short = QtCore.Signal(str, str, object, str)
    status = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self, factory: PipelineFactory, producer: AudioProducer):
        super().__init__()
        self._factory = factory
        self._producer = producer
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._work_q_short: Queue = Queue()
        self._translate_enabled = True
        self._short_enabled = True
        self._pipeline_lock = threading.Lock()
        self._pipeline: Optional[SpeechToEnglishPipeline] = None
        self._cfg: Optional[RunConfig] = None
        self._translator_lock = threading.Lock()
        self._translator_cache: dict[str, Translator] = {}
        self._source_lang_override: Optional[str] = None
        self._chunk_seconds = 1.0

    def set_translation_enabled(self, enabled: bool):
        self._translate_enabled = enabled

    def _is_translation_enabled(self) -> bool:
        return self._translate_enabled

    def set_short_enabled(self, enabled: bool):
        self._short_enabled = enabled

    def _is_short_enabled(self) -> bool:
        return self._short_enabled

    def start(self, cfg: RunConfig):
        if self._threads:
            return
        self._cfg = cfg
        self._chunk_seconds = cfg.chunk_seconds
        self._stop.clear()
        self.status.emit("Starting...")
        threading.Thread(target=self._init_and_run, args=(cfg,), daemon=True).start()

    def stop(self):
        if not self._threads:
            return
        self.status.emit("Stopping...")
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads = []
        with self._pipeline_lock:
            self._pipeline = None
        self._cfg = None
        self._source_lang_override = None
        self.status.emit("Idle")

    def _init_and_run(self, cfg: RunConfig):
        try:
            pipeline_short = self._factory.create(cfg)
            with self._pipeline_lock:
                self._pipeline = pipeline_short
            vad = SileroVAD()
        except Exception as exc:
            self.error.emit(str(exc))
            self.status.emit("Idle")
            return

        out_short = self._work_q_short
        producer_t = threading.Thread(
            target=self._producer.run,
            args=(
                self._stop,
                out_short,
                pipeline_short.recognizer.target_sr,
                self._is_short_enabled,
                vad,
                self._get_chunk_seconds,
                cfg.use_loopback,
            ),
            daemon=True,
        )
        short_t = threading.Thread(
            target=ShortProcessor(
                self._get_pipeline, self._get_source_lang, self._get_translator
            ).run,
            args=(
                self._stop,
                self._work_q_short,
                self.text_ready_short.emit,
                self._is_translation_enabled,
            ),
            daemon=True,
        )
        self._threads = [producer_t, short_t]
        for t in self._threads:
            t.start()
        self.status.emit("Listening")

    def _get_pipeline(self) -> Optional[SpeechToEnglishPipeline]:
        with self._pipeline_lock:
            return self._pipeline

    def _get_source_lang(self) -> str:
        cfg = self._cfg
        if self._source_lang_override is not None:
            return self._source_lang_override
        return cfg.source_lang if cfg is not None else "ja"

    def _get_translator(self, source_lang: str) -> Translator:
        key = "ja" if source_lang == "ja" else "en"
        with self._translator_lock:
            if key in self._translator_cache:
                return self._translator_cache[key]
            translator = self._factory.create_translator(key)
            self._translator_cache[key] = translator
            return translator

    def set_source_lang(self, source_lang: str):
        cfg = self._cfg
        if cfg is None:
            return
        if cfg.source_lang == source_lang and self._source_lang_override is None:
            return
        self._source_lang_override = source_lang
        with self._pipeline_lock:
            pipeline = self._pipeline
            if pipeline is not None and hasattr(pipeline.recognizer, "language"):
                setattr(pipeline.recognizer, "language", source_lang)

    def set_chunk_seconds(self, chunk_seconds: float):
        if chunk_seconds <= 0:
            return
        self._chunk_seconds = float(chunk_seconds)

    def _get_chunk_seconds(self) -> float:
        return self._chunk_seconds
