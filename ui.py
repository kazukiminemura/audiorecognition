import sys
import threading
from queue import Queue, Empty

from PySide6 import QtCore, QtGui, QtWidgets

from pipeline import SpeechToEnglishPipeline
from pipeline_impl import JaToEnTranslator
from transcribe import WhisperOVTranscriber, record_audio


class Worker(QtCore.QObject):
    text_ready = QtCore.Signal(str, str)
    status = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self._stop = threading.Event()
        self._work_q: Queue = Queue()
        self._threads: list[threading.Thread] = []
        self._pipeline: SpeechToEnglishPipeline | None = None
        self._use_loopback = False

    def start(self, use_loopback: bool):
        if self._threads:
            return
        self._use_loopback = use_loopback
        self._stop.clear()
        transcriber = WhisperOVTranscriber(language="ja", task="transcribe")
        self._pipeline = SpeechToEnglishPipeline(transcriber, JaToEnTranslator())
        self._threads = [
            threading.Thread(target=self._producer, daemon=True),
            threading.Thread(target=self._consumer, daemon=True),
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
        self._pipeline = None
        self.status.emit("Idle")

    def _producer(self):
        assert self._pipeline is not None
        transcriber = self._pipeline.recognizer
        while not self._stop.is_set():
            audio, sr = record_audio(
                duration_s=4.0,
                target_sr=transcriber.target_sr,
                loopback=self._use_loopback,
            )
            self._work_q.put((audio, sr))

    def _consumer(self):
        assert self._pipeline is not None
        while not self._stop.is_set() or not self._work_q.empty():
            try:
                audio, sr = self._work_q.get(timeout=0.5)
            except Empty:
                continue
            jp = self._pipeline.recognizer.transcribe_array(audio, sr)
            en = self._pipeline.translator.translate(jp) if jp else ""
            self.text_ready.emit(jp, en)
            self._work_q.task_done()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech -> English")
        self.resize(920, 680)

        self.worker = Worker()
        self.worker.text_ready.connect(self._append_text)
        self.worker.status.connect(self._set_status)

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
        self.status_lbl = QtWidgets.QLabel("Idle")
        self.status_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addSpacing(8)
        btn_row.addWidget(self.system_audio)
        btn_row.addStretch(1)
        btn_row.addWidget(self.status_lbl)

        jp_label = QtWidgets.QLabel("Japanese (recognized)")
        jp_label.setObjectName("Section")
        self.jp_text = QtWidgets.QPlainTextEdit()
        self.jp_text.setReadOnly(True)

        en_label = QtWidgets.QLabel("English (translated)")
        en_label.setObjectName("Section")
        self.en_text = QtWidgets.QPlainTextEdit()
        self.en_text.setReadOnly(True)

        layout.addWidget(title)
        layout.addLayout(btn_row)
        layout.addWidget(jp_label)
        layout.addWidget(self.jp_text, 1)
        layout.addWidget(en_label)
        layout.addWidget(self.en_text, 1)

        self._apply_style()

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
        self.worker.start(use_loopback=self.system_audio.isChecked())

    def _stop(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker.stop()

    @QtCore.Slot(str, str)
    def _append_text(self, jp: str, en: str):
        if jp:
            self.jp_text.appendPlainText(jp)
        if en:
            self.en_text.appendPlainText(en)

    @QtCore.Slot(str)
    def _set_status(self, status: str):
        self.status_lbl.setText(status)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
