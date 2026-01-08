import threading
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Callable, Optional

import numpy as np
from PySide6 import QtCore

from diarization import PyannoteDiarizer
from pipeline import SpeechToEnglishPipeline
from pipeline_impl import JaToEnTranslator, EnToJaTranslator
from whisper_transcriber import WhisperOVTranscriber


@dataclass(frozen=True)
class RunConfig:
    use_loopback: bool
    source_lang: str
    chunk_seconds: float
    enable_short: bool
    enable_translation: bool
    enable_diarization: bool
    diarization_speakers: int


class PipelineFactory:
    def create(self, source_lang: str) -> SpeechToEnglishPipeline:
        transcriber = WhisperOVTranscriber(
            language=source_lang,
            task="transcribe",
            device="GPU",
        )
        translator = JaToEnTranslator() if source_lang == "ja" else EnToJaTranslator()
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
    ):
        while not stop_event.is_set():
            audio, sr = self._record_fn(
                duration_s=cfg.chunk_seconds,
                target_sr=target_sr,
                loopback=cfg.use_loopback,
            )
            if audio.size == 0:
                continue
            if out_short is not None and enable_short():
                out_short.put((audio, sr))


class ShortProcessor:
    def __init__(self, pipeline: SpeechToEnglishPipeline, diarizer: Optional[PyannoteDiarizer]):
        self._pipeline = pipeline
        self._diarizer = diarizer

    def run(
        self,
        stop_event: threading.Event,
        in_q: Queue,
        emit: Callable[[str, str, Optional[str]], None],
        enable_translation: Callable[[], bool],
        enable_diarization: Callable[[], bool],
        get_diarization_speakers: Callable[[], int],
    ):
        while not stop_event.is_set() or not in_q.empty():
            try:
                audio, sr = in_q.get(timeout=0.5)
            except Empty:
                continue
            if audio.size == 0:
                in_q.task_done()
                continue
            jp = self._pipeline.recognizer.transcribe_array(audio, sr)
            en = self._pipeline.translator.translate(jp) if jp and enable_translation() else ""
            num_speakers = get_diarization_speakers()
            speaker = (
                self._diarizer.dominant_speaker(
                    audio,
                    sr,
                    num_speakers=num_speakers if num_speakers > 0 else None,
                )
                if self._diarizer is not None and enable_diarization()
                else None
            )
            emit(jp, en, speaker)
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
        self._diarization_enabled = True
        self._diarization_speakers = 0

    def set_translation_enabled(self, enabled: bool):
        self._translate_enabled = enabled

    def _is_translation_enabled(self) -> bool:
        return self._translate_enabled

    def set_short_enabled(self, enabled: bool):
        self._short_enabled = enabled

    def _is_short_enabled(self) -> bool:
        return self._short_enabled

    def set_diarization_enabled(self, enabled: bool):
        self._diarization_enabled = enabled

    def _is_diarization_enabled(self) -> bool:
        return self._diarization_enabled

    def set_diarization_speakers(self, count: int):
        self._diarization_speakers = max(0, int(count))

    def _get_diarization_speakers(self) -> int:
        return self._diarization_speakers

    def start(self, cfg: RunConfig):
        if self._threads:
            return
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
        self.status.emit("Idle")

    def _init_and_run(self, cfg: RunConfig):
        try:
            pipeline_short = self._factory.create(cfg.source_lang)
            diarizer = PyannoteDiarizer() if cfg.enable_diarization else None
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
            ),
            daemon=True,
        )
        short_t = threading.Thread(
            target=ShortProcessor(pipeline_short, diarizer).run,
            args=(
                self._stop,
                self._work_q_short,
                self.text_ready_short.emit,
                self._is_translation_enabled,
                self._is_diarization_enabled,
                self._get_diarization_speakers,
            ),
            daemon=True,
        )
        self._threads = [producer_t, short_t]
        for t in self._threads:
            t.start()
        self.status.emit("Listening")
