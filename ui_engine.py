import threading
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Callable, Optional

import numpy as np
from PySide6 import QtCore

from pipeline import SpeechToEnglishPipeline
from pipeline_impl import JaToEnTranslator, EnToJaTranslator
from lfm2_transcriber import LFM2AudioTranscriber
from whisper_transcriber import WhisperOVTranscriber
from vad import SileroVAD


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


class AudioProducer:
    def __init__(self, record_fn: Callable[..., tuple[np.ndarray, int]]):
        self._record_fn = record_fn

    def run(
        self,
        stop_event: threading.Event,
        cfg: RunConfig,
        out_short: Optional[Queue],
        target_sr: int,
        enable_short: Callable[[], bool],
        vad: Optional[SileroVAD],
    ):
        while not stop_event.is_set():
            audio, sr = self._record_fn(
                duration_s=cfg.chunk_seconds,
                target_sr=target_sr,
                loopback=cfg.use_loopback,
            )
            if audio.size == 0:
                continue
            if vad is not None and not vad.has_speech(audio, sr):
                continue
            if out_short is not None and enable_short():
                out_short.put((audio, sr))


class ShortProcessor:
    def __init__(self, get_pipeline: Callable[[], Optional[SpeechToEnglishPipeline]]):
        self._get_pipeline = get_pipeline

    def run(
        self,
        stop_event: threading.Event,
        in_q: Queue,
        emit: Callable[[str, str, Optional[str]], None],
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
            jp = pipeline.recognizer.transcribe_array(audio, sr)
            en = pipeline.translator.translate(jp) if jp and enable_translation() else ""
            emit(jp, en, None)
            in_q.task_done()


class SpeechEngine(QtCore.QObject):
    text_ready_short = QtCore.Signal(str, str, object)
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
                cfg,
                out_short,
                pipeline_short.recognizer.target_sr,
                self._is_short_enabled,
                vad,
            ),
            daemon=True,
        )
        short_t = threading.Thread(
            target=ShortProcessor(self._get_pipeline).run,
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

    def set_source_lang(self, source_lang: str):
        cfg = self._cfg
        if cfg is None:
            return
        if cfg.source_lang == source_lang:
            return
        new_cfg = RunConfig(
            use_loopback=cfg.use_loopback,
            source_lang=source_lang,
            chunk_seconds=cfg.chunk_seconds,
            engine=cfg.engine,
            lfm2_repo=cfg.lfm2_repo,
            enable_short=cfg.enable_short,
            enable_translation=cfg.enable_translation,
        )
        try:
            new_pipeline = self._factory.create(new_cfg)
        except Exception as exc:
            self.error.emit(str(exc))
            return
        with self._pipeline_lock:
            self._pipeline = new_pipeline
        self._cfg = new_cfg
