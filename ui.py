import sys
import threading
import math
import time
from collections import deque
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Callable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from pipeline import SpeechToEnglishPipeline
from pipeline_impl import JaToEnTranslator, EnToJaTranslator
from transcribe import WhisperOVTranscriber, record_audio


@dataclass(frozen=True)
class RunConfig:
    use_loopback: bool
    source_lang: str
    chunk_seconds: float


class PipelineFactory:
    def create(self, source_lang: str) -> SpeechToEnglishPipeline:
        transcriber = WhisperOVTranscriber(language=source_lang, task="transcribe")
        translator = JaToEnTranslator() if source_lang == "ja" else EnToJaTranslator()
        return SpeechToEnglishPipeline(transcriber, translator)


class AudioProducer:
    def __init__(self, record_fn: Callable[..., tuple[np.ndarray, int]]):
        self._record_fn = record_fn

    def run(self, stop_event: threading.Event, cfg: RunConfig, out_short: Queue, out_long: Queue, target_sr: int):
        while not stop_event.is_set():
            audio, sr = self._record_fn(
                duration_s=cfg.chunk_seconds,
                target_sr=target_sr,
                loopback=cfg.use_loopback,
            )
            out_short.put((audio, sr))
            out_long.put((audio, sr))


class ShortProcessor:
    def __init__(self, pipeline: SpeechToEnglishPipeline):
        self._pipeline = pipeline

    def run(self, stop_event: threading.Event, in_q: Queue, emit: Callable[[str, str], None]):
        while not stop_event.is_set() or not in_q.empty():
            try:
                audio, sr = in_q.get(timeout=0.5)
            except Empty:
                continue
            jp = self._pipeline.recognizer.transcribe_array(audio, sr)
            en = self._pipeline.translator.translate(jp) if jp else ""
            emit(jp, en)
            in_q.task_done()


class LongProcessor:
    def __init__(self, pipeline: SpeechToEnglishPipeline, concat_chunks: int):
        self._pipeline = pipeline
        self._concat_chunks = concat_chunks
        self._buffer: deque[np.ndarray] = deque()

    def run(self, stop_event: threading.Event, in_q: Queue, emit: Callable[[str, str, bool], None]):
        while not stop_event.is_set() or not in_q.empty():
            try:
                audio, sr = in_q.get(timeout=0.5)
            except Empty:
                continue
            self._buffer.append(audio)
            while len(self._buffer) > self._concat_chunks:
                self._buffer.popleft()
            concat_audio = np.concatenate(list(self._buffer), axis=0)
            jp = self._pipeline.recognizer.transcribe_array(concat_audio, sr)
            en = self._pipeline.translator.translate(jp) if jp else ""
            replace = len(self._buffer) > 1
            emit(jp, en, replace)
            in_q.task_done()


class SpeechEngine(QtCore.QObject):
    text_ready_short = QtCore.Signal(str, str)
    text_ready_long = QtCore.Signal(str, str, bool)
    status = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self, factory: PipelineFactory, producer: AudioProducer):
        super().__init__()
        self._factory = factory
        self._producer = producer
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._work_q_short: Queue = Queue()
        self._work_q_long: Queue = Queue()

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
            pipeline_long = self._factory.create(cfg.source_lang)
        except Exception as exc:
            self.error.emit(str(exc))
            self.status.emit("Idle")
            return

        window_seconds = max(6.0, cfg.chunk_seconds * 2)
        concat_chunks = max(2, int(math.ceil(window_seconds / cfg.chunk_seconds)))

        producer_t = threading.Thread(
            target=self._producer.run,
            args=(
                self._stop,
                cfg,
                self._work_q_short,
                self._work_q_long,
                pipeline_short.recognizer.target_sr,
            ),
            daemon=True,
        )
        short_t = threading.Thread(
            target=ShortProcessor(pipeline_short).run,
            args=(self._stop, self._work_q_short, self.text_ready_short.emit),
            daemon=True,
        )
        long_t = threading.Thread(
            target=LongProcessor(pipeline_long, concat_chunks).run,
            args=(self._stop, self._work_q_long, self.text_ready_long.emit),
            daemon=True,
        )
        self._threads = [producer_t, short_t, long_t]
        for t in self._threads:
            t.start()
        self.status.emit("Listening")


class TextAccumulator:
    def __init__(self):
        self._segments: list[str] = []

    def append(self, text: str):
        self._segments.append(text)

    def replace_last(self, text: str):
        if self._segments:
            self._segments[-1] = text
        else:
            self._segments.append(text)

    def render(self) -> str:
        return "\n".join(self._segments)


class LongUpdatePolicy:
    def __init__(self):
        self._last_update_ts = 0.0
        self._update_count = 0

    def should_newline(self, replace: bool) -> bool:
        now = time.monotonic()
        should_newline = False
        if replace:
            self._update_count += 1
            if (now - self._last_update_ts) > 3.0 or self._update_count >= 3:
                should_newline = True
        else:
            self._update_count = 0
        self._last_update_ts = now
        return should_newline


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, engine: SpeechEngine):
        super().__init__()
        self.setWindowTitle("Speech -> English")
        self.resize(920, 680)

        self._engine = engine
        self._engine.text_ready_short.connect(self._append_short)
        self._engine.text_ready_long.connect(self._append_long)
        self._engine.status.connect(self._set_status)
        self._engine.error.connect(self._show_error)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("Speech Recognition + Translation")
        title.setObjectName("Title")

        btn_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.system_audio = QtWidgets.QCheckBox("System Audio")
        self.lang_select = QtWidgets.QComboBox()
        self.lang_select.addItems(["Japanese -> English", "English -> Japanese"])
        self.chunk_spin = QtWidgets.QDoubleSpinBox()
        self.chunk_spin.setRange(0.5, 10.0)
        self.chunk_spin.setSingleStep(0.5)
        self.chunk_spin.setValue(1.0)
        self.chunk_spin.setSuffix(" s")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(10)
        self.progress.setVisible(False)
        self.status_lbl = QtWidgets.QLabel("Idle")
        self.status_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addSpacing(8)
        btn_row.addWidget(self.system_audio)
        btn_row.addSpacing(8)
        btn_row.addWidget(self.lang_select)
        btn_row.addSpacing(8)
        btn_row.addWidget(QtWidgets.QLabel("Chunk"))
        btn_row.addWidget(self.chunk_spin)
        btn_row.addStretch(1)
        btn_row.addWidget(self.progress)
        btn_row.addSpacing(8)
        btn_row.addWidget(self.status_lbl)

        jp_label = QtWidgets.QLabel("Short (single chunk) - Japanese")
        jp_label.setObjectName("Section")
        self.jp_text = QtWidgets.QPlainTextEdit()
        self.jp_text.setReadOnly(True)

        en_label = QtWidgets.QLabel("Short (single chunk) - English")
        en_label.setObjectName("Section")
        self.en_text = QtWidgets.QPlainTextEdit()
        self.en_text.setReadOnly(True)

        jp_long_label = QtWidgets.QLabel("Long (concatenated) - Japanese")
        jp_long_label.setObjectName("Section")
        self.jp_long_text = QtWidgets.QPlainTextEdit()
        self.jp_long_text.setReadOnly(True)

        en_long_label = QtWidgets.QLabel("Long (concatenated) - English")
        en_long_label.setObjectName("Section")
        self.en_long_text = QtWidgets.QPlainTextEdit()
        self.en_long_text.setReadOnly(True)

        layout.addWidget(title)
        layout.addLayout(btn_row)
        layout.addWidget(jp_label)
        layout.addWidget(self.jp_text, 1)
        layout.addWidget(en_label)
        layout.addWidget(self.en_text, 1)
        layout.addWidget(jp_long_label)
        layout.addWidget(self.jp_long_text, 1)
        layout.addWidget(en_long_label)
        layout.addWidget(self.en_long_text, 1)

        self._apply_style()
        self._short_jp = TextAccumulator()
        self._short_en = TextAccumulator()
        self._long_jp = TextAccumulator()
        self._long_en = TextAccumulator()
        self._long_policy = LongUpdatePolicy()

    def closeEvent(self, event):
        self._engine.stop()
        event.accept()

    def _apply_style(self):
        self.setStyleSheet(
            """
            QWidget { font-family: "Segoe UI"; font-size: 11pt; }
            QMainWindow { background: #f6f8fb; }
            QLabel#Title { font-size: 16pt; font-weight: 600; color: #0b1f33; }
            QLabel#Section { font-size: 9pt; font-weight: 600; color: #5b6b7a; }
            QPushButton {
                background: #0078d4; color: white; border: none;
                padding: 6px 14px; border-radius: 6px;
            }
            QPushButton:disabled { background: #c9d1d9; color: #5b6b7a; }
            QProgressBar {
                border: 1px solid #d6dde6; border-radius: 5px;
                background: #eef2f7;
            }
            QProgressBar::chunk { background: #0078d4; border-radius: 5px; }
            QPlainTextEdit {
                background: white; border: 1px solid #d6dde6; border-radius: 6px;
                padding: 6px;
            }
            QCheckBox { color: #0b1f33; }
            """
        )

    def _start(self):
        source_lang = "ja" if self.lang_select.currentIndex() == 0 else "en"
        cfg = RunConfig(
            use_loopback=self.system_audio.isChecked(),
            source_lang=source_lang,
            chunk_seconds=self.chunk_spin.value(),
        )
        self._engine.start(cfg)

    def _stop(self):
        self._engine.stop()

    @QtCore.Slot(str, str)
    def _append_short(self, jp: str, en: str):
        if jp:
            self._short_jp.append(jp)
            self.jp_text.setPlainText(self._short_jp.render())
            self.jp_text.moveCursor(QtGui.QTextCursor.End)
        if en:
            self._short_en.append(en)
            self.en_text.setPlainText(self._short_en.render())
            self.en_text.moveCursor(QtGui.QTextCursor.End)

    @QtCore.Slot(str, str, bool)
    def _append_long(self, jp: str, en: str, replace: bool):
        should_newline = self._long_policy.should_newline(replace)
        if jp:
            if replace and not should_newline:
                self._long_jp.replace_last(jp)
            else:
                self._long_jp.append(jp)
            self.jp_long_text.setPlainText(self._long_jp.render())
            self.jp_long_text.moveCursor(QtGui.QTextCursor.End)
        if en:
            if replace and not should_newline:
                self._long_en.replace_last(en)
            else:
                self._long_en.append(en)
            self.en_long_text.setPlainText(self._long_en.render())
            self.en_long_text.moveCursor(QtGui.QTextCursor.End)

    @QtCore.Slot(str)
    def _set_status(self, status: str):
        self.status_lbl.setText(status)
        if status == "Listening":
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress.setVisible(False)
        elif status in ("Idle", "Stopping..."):
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress.setVisible(False)
            self.system_audio.setEnabled(True)
            self.lang_select.setEnabled(True)
            self.chunk_spin.setEnabled(True)
        elif status == "Starting...":
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress.setVisible(True)

    @QtCore.Slot(str)
    def _show_error(self, message: str):
        QtWidgets.QMessageBox.critical(self, "Error", message)


def main():
    app = QtWidgets.QApplication(sys.argv)
    engine = SpeechEngine(PipelineFactory(), AudioProducer(record_audio))
    window = MainWindow(engine)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
