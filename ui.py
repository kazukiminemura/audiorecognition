import sys

from PySide6 import QtWidgets

from audio_io import record_audio
from ui_engine import AudioProducer, PipelineFactory, SpeechEngine
from ui_window import MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    engine = SpeechEngine(PipelineFactory(), AudioProducer(record_audio))
    window = MainWindow(engine)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
