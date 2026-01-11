from src import hf_env  # ensure HF env defaults before other imports
import os
import sys

from PySide6 import QtCore, QtWidgets

from src.audio_io import record_audio
from src.engine.ui_engine import AudioProducer, PipelineFactory, SpeechEngine
from src.ui.window import MainWindow


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

        proc = QtCore.QProcess(window)
        proc.setProgram(sys.executable)
        proc.setArguments(["download_models.py"])
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        proc.setWorkingDirectory(project_root)

        def _finish():
            dialog.close()

        def _error(_err):
            dialog.close()
            QtWidgets.QMessageBox.critical(
                window, "Error", "Model download process failed."
            )

        proc.finished.connect(_finish)
        proc.errorOccurred.connect(_error)
        dialog.canceled.connect(proc.kill)

        proc.start()
        dialog.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
