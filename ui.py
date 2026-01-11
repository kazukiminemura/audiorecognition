import hf_env  # ensure HF env defaults before other imports
import os
import sys

from PySide6 import QtCore, QtWidgets

from audio_io import record_audio
from ui_engine import AudioProducer, PipelineFactory, SpeechEngine
from ui_window import MainWindow
from model_preload import preload_models


class PreloadWorker(QtCore.QObject):
    progress = QtCore.Signal(str)
    finished = QtCore.Signal(bool)
    error = QtCore.Signal(str)

    @QtCore.Slot()
    def run(self):
        try:
            ok = preload_models(
                progress_cb=self.progress.emit,
                is_cancelled=QtCore.QThread.currentThread().isInterruptionRequested,
            )
            self.finished.emit(ok)
        except Exception as exc:
            self.error.emit(str(exc))


def main():
    app = QtWidgets.QApplication(sys.argv)
    engine = SpeechEngine(PipelineFactory(), AudioProducer(record_audio))
    window = MainWindow(engine)

    window.show()

    if os.getenv("PRELOAD_ON_START", "1") not in ("0", "false", "False", "no", "NO"):
        dialog = QtWidgets.QProgressDialog(
            "Preparing models...", "Cancel", 0, 0, window
        )
        dialog.setWindowTitle("Downloading models")
        dialog.setWindowModality(QtCore.Qt.WindowModal)
        dialog.setMinimumDuration(0)

        thread = QtCore.QThread()
        worker = PreloadWorker()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(dialog.setLabelText)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(thread.quit)
        worker.error.connect(worker.deleteLater)

        def _finish(ok: bool):
            dialog.close()

        worker.finished.connect(_finish)
        worker.error.connect(
            lambda msg: QtWidgets.QMessageBox.critical(window, "Error", msg)
        )
        thread.finished.connect(thread.deleteLater)

        dialog.canceled.connect(thread.requestInterruption)
        dialog.canceled.connect(thread.quit)

        thread.start()
        dialog.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
