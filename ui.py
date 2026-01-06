import sys
import threading
import math
from collections import deque
from queue import Queue, Empty

import numpy as np
import time

from PySide6 import QtCore, QtGui, QtWidgets

from pipeline import SpeechToEnglishPipeline
from pipeline_impl import JaToEnTranslator, EnToJaTranslator
from transcribe import WhisperOVTranscriber, record_audio


class Worker(QtCore.QObject):
    text_ready_short = QtCore.Signal(str, str)
    text_ready_long = QtCore.Signal(str, str, bool)
    status = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self._stop = threading.Event()
        self._work_q_short: Queue = Queue()
        self._work_q_long: Queue = Queue()
        self._threads: list[threading.Thread] = []
        self._pipeline_short: SpeechToEnglishPipeline | None = None
        self._pipeline_long: SpeechToEnglishPipeline | None = None
        self._use_loopback = False
        self._chunk_seconds = 2.5
        self._concat_chunks = 3
        self._audio_buf: deque[np.ndarray] = deque()
        self._init_thread: threading.Thread | None = None

    def start(self, use_loopback: bool, source_lang: str, chunk_seconds: float):
        if self._threads:
            return
        self._use_loopback = use_loopback
        self._chunk_seconds = float(chunk_seconds)
        window_seconds = max(6.0, self._chunk_seconds * 2)
        self._concat_chunks = max(2, int(math.ceil(window_seconds / self._chunk_seconds)))
        self._audio_buf.clear()
        self._stop.clear()
        self.status.emit("Starting...")
        self._init_thread = threading.Thread(
            target=self._init_and_run, args=(source_lang,), daemon=True
        )
        self._init_thread.start()

    def _init_and_run(self, source_lang: str):
        try:
            transcriber_short = WhisperOVTranscriber(
                language=source_lang, task="transcribe"
            )
            transcriber_long = WhisperOVTranscriber(
                language=source_lang, task="transcribe"
            )
            translator_short = (
                JaToEnTranslator() if source_lang == "ja" else EnToJaTranslator()
            )
            translator_long = (
                JaToEnTranslator() if source_lang == "ja" else EnToJaTranslator()
            )
            self._pipeline_short = SpeechToEnglishPipeline(
                transcriber_short, translator_short
            )
            self._pipeline_long = SpeechToEnglishPipeline(
                transcriber_long, translator_long
            )
        except Exception as exc:
            self.error.emit(str(exc))
            self.status.emit("Idle")
            return
        self._threads = [
            threading.Thread(target=self._producer, daemon=True),
            threading.Thread(target=self._consumer_short, daemon=True),
            threading.Thread(target=self._consumer_long, daemon=True),
        ]
        for t in self._threads:
            t.start()
        self.status.emit("Listening")

    def stop(self):
        if not self._threads:
            return
        self.status.emit("Stopping...")
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads = []
        self._pipeline_short = None
        self._pipeline_long = None
        self.status.emit("Idle")

    def _producer(self):
        pipeline_short = self._pipeline_short
        if pipeline_short is None:
            return
        transcriber = pipeline_short.recognizer
        while not self._stop.is_set():
            audio, sr = record_audio(
                duration_s=self._chunk_seconds,
                target_sr=transcriber.target_sr,
                loopback=self._use_loopback,
            )
            self._work_q_short.put((audio, sr))
            self._work_q_long.put((audio, sr))

    def _consumer_short(self):
        while not self._stop.is_set() or not self._work_q_short.empty():
            pipeline = self._pipeline_short
            if pipeline is None:
                break
            try:
                audio, sr = self._work_q_short.get(timeout=0.5)
            except Empty:
                continue
            jp_short = pipeline.recognizer.transcribe_array(audio, sr)
            en_short = pipeline.translator.translate(jp_short) if jp_short else ""
            self.text_ready_short.emit(jp_short, en_short)
            self._work_q_short.task_done()

    def _consumer_long(self):
        while not self._stop.is_set() or not self._work_q_long.empty():
            pipeline = self._pipeline_long
            if pipeline is None:
                break
            try:
                audio, sr = self._work_q_long.get(timeout=0.5)
            except Empty:
                continue
            self._audio_buf.append(audio)
            while len(self._audio_buf) > self._concat_chunks:
                self._audio_buf.popleft()
            concat_audio = np.concatenate(list(self._audio_buf), axis=0)
            jp = pipeline.recognizer.transcribe_array(concat_audio, sr)
            en = pipeline.translator.translate(jp) if jp else ""
            replace = len(self._audio_buf) > 1
            self.text_ready_long.emit(jp, en, replace)
            self._work_q_long.task_done()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech -> English")
        self.resize(920, 680)

        self.worker = Worker()
        self.worker.text_ready_short.connect(self._append_short)
        self.worker.text_ready_long.connect(self._append_long)
        self.worker.status.connect(self._set_status)
        self.worker.error.connect(self._show_error)

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
        self.chunk_spin.setValue(2.5)
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
        self._jp_segments: list[str] = []
        self._en_segments: list[str] = []
        self._jp_long_segments: list[str] = []
        self._en_long_segments: list[str] = []
        self._last_update_ts = 0.0
        self._update_count = 0

    def closeEvent(self, event):
        self.worker.stop()
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
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_lbl.setText("Starting...")
        self.system_audio.setEnabled(False)
        self.lang_select.setEnabled(False)
        self.chunk_spin.setEnabled(False)
        self.progress.setVisible(True)
        source_lang = "ja" if self.lang_select.currentIndex() == 0 else "en"
        self.worker.start(
            use_loopback=self.system_audio.isChecked(),
            source_lang=source_lang,
            chunk_seconds=self.chunk_spin.value(),
        )

    def _stop(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker.stop()

    @QtCore.Slot(str, str)
    def _append_short(self, jp: str, en: str):
        if jp:
            self._jp_segments.append(jp)
            self.jp_text.setPlainText("\n".join(self._jp_segments))
            self.jp_text.moveCursor(QtGui.QTextCursor.End)
        if en:
            self._en_segments.append(en)
            self.en_text.setPlainText("\n".join(self._en_segments))
            self.en_text.moveCursor(QtGui.QTextCursor.End)

    @QtCore.Slot(str, str, bool)
    def _append_long(self, jp: str, en: str, replace: bool):
        now = time.monotonic()
        should_newline = False
        if replace:
            self._update_count += 1
            if (now - self._last_update_ts) > 3.0 or self._update_count >= 3:
                should_newline = True
        else:
            self._update_count = 0
        self._last_update_ts = now

        if jp:
            if (replace and self._jp_long_segments) and not should_newline:
                self._jp_long_segments[-1] = jp
            else:
                self._jp_long_segments.append(jp)
            self.jp_long_text.setPlainText("\n".join(self._jp_long_segments))
            self.jp_long_text.moveCursor(QtGui.QTextCursor.End)
        if en:
            if (replace and self._en_long_segments) and not should_newline:
                self._en_long_segments[-1] = en
            else:
                self._en_long_segments.append(en)
            self.en_long_text.setPlainText("\n".join(self._en_long_segments))
            self.en_long_text.moveCursor(QtGui.QTextCursor.End)

    @QtCore.Slot(str)
    def _set_status(self, status: str):
        self.status_lbl.setText(status)
        if status == "Listening":
            self.progress.setVisible(False)
        elif status in ("Idle", "Stopping..."):
            self.progress.setVisible(False)
            self.system_audio.setEnabled(True)
            self.lang_select.setEnabled(True)
            self.chunk_spin.setEnabled(True)
        elif status == "Starting...":
            self.progress.setVisible(True)

    @QtCore.Slot(str)
    def _show_error(self, message: str):
        QtWidgets.QMessageBox.critical(self, "Error", message)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setVisible(False)
        self.system_audio.setEnabled(True)
        self.lang_select.setEnabled(True)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
