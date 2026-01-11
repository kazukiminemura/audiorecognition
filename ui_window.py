import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from settings import LFM2_AUDIO_REPO
from ui_engine import RunConfig, SpeechEngine
from ui_models import SpeakerTextAccumulator, SpeakerPalette


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, engine: SpeechEngine):
        super().__init__()
        self.setWindowTitle("Speech -> English")
        icon_path = _resource_path(Path("deployment") / "favicon.ico")
        if icon_path.exists():
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))
        self.resize(920, 680)

        self._engine = engine
        self._engine.text_ready_short.connect(self._append_short)
        self._engine.status.connect(self._set_status)
        self._engine.error.connect(self._show_error)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("Speech Recognition + Translation")
        title.setObjectName("Title")

        btn_row_top = QtWidgets.QHBoxLayout()
        btn_row_bottom = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.save_btn = QtWidgets.QPushButton("Save Script")
        self.reset_btn = QtWidgets.QPushButton("Reset Text")
        self.stop_btn.setEnabled(False)
        self.system_audio = QtWidgets.QCheckBox("System Audio")
        self.lang_select = QtWidgets.QComboBox()
        self.lang_select.addItems(["Japanese -> English", "English -> Japanese"])
        self.engine_select = QtWidgets.QComboBox()
        self.engine_select.addItems(["Whisper (OpenVINO)", "LiquidAI LFM2.5-Audio (CPU)"])
        self.lfm2_repo = QtWidgets.QLineEdit(LFM2_AUDIO_REPO)
        self.lfm2_repo.setPlaceholderText("Hugging Face repo id")
        self.chunk_spin = QtWidgets.QDoubleSpinBox()
        self.chunk_spin.setRange(0.5, 10.0)
        self.chunk_spin.setSingleStep(0.5)
        self.chunk_spin.setValue(1.0)
        self.chunk_spin.setSuffix(" s")
        self.short_toggle = QtWidgets.QCheckBox("Short")
        self.short_toggle.setChecked(True)
        self.translate_toggle = QtWidgets.QCheckBox("Translate")
        self.translate_toggle.setChecked(True)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(10)
        self.progress.setVisible(False)
        self.status_lbl = QtWidgets.QLabel("Idle")
        self.status_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.save_btn.clicked.connect(self._save_script)
        self.reset_btn.clicked.connect(self._reset_text)
        self.lang_select.currentIndexChanged.connect(self._set_source_lang)
        self.engine_select.currentIndexChanged.connect(self._toggle_engine_fields)
        self.short_toggle.toggled.connect(self._engine.set_short_enabled)
        self.translate_toggle.toggled.connect(self._engine.set_translation_enabled)

        btn_row_top.addWidget(self.start_btn)
        btn_row_top.addWidget(self.stop_btn)
        btn_row_top.addSpacing(4)
        btn_row_top.addWidget(self.save_btn)
        btn_row_top.addWidget(self.reset_btn)
        btn_row_top.addStretch(1)
        btn_row_top.addWidget(self.progress)
        btn_row_top.addSpacing(8)
        btn_row_top.addWidget(self.status_lbl)

        btn_row_bottom.addWidget(self.system_audio)
        btn_row_bottom.addSpacing(8)
        btn_row_bottom.addWidget(self.lang_select)
        btn_row_bottom.addSpacing(8)
        btn_row_bottom.addWidget(QtWidgets.QLabel("Chunk"))
        btn_row_bottom.addWidget(self.chunk_spin)
        btn_row_bottom.addSpacing(8)
        btn_row_bottom.addWidget(self.short_toggle)
        btn_row_bottom.addWidget(self.translate_toggle)
        btn_row_bottom.addStretch(1)

        btn_row_engine = QtWidgets.QHBoxLayout()
        btn_row_engine.addWidget(QtWidgets.QLabel("Engine"))
        btn_row_engine.addWidget(self.engine_select)
        btn_row_engine.addSpacing(8)
        btn_row_engine.addWidget(QtWidgets.QLabel("LFM2 Repo"))
        btn_row_engine.addWidget(self.lfm2_repo)
        btn_row_engine.addStretch(1)

        jp_label = QtWidgets.QLabel("Short (single chunk) - Japanese")
        jp_label.setObjectName("Section")
        self.jp_text = QtWidgets.QTextEdit()
        self.jp_text.setReadOnly(True)

        en_label = QtWidgets.QLabel("Short (single chunk) - English")
        en_label.setObjectName("Section")
        self.en_text = QtWidgets.QTextEdit()
        self.en_text.setReadOnly(True)

        layout.addWidget(title)
        layout.addLayout(btn_row_top)
        layout.addLayout(btn_row_bottom)
        layout.addLayout(btn_row_engine)
        layout.addWidget(jp_label)
        layout.addWidget(self.jp_text, 1)
        layout.addWidget(en_label)
        layout.addWidget(self.en_text, 1)

        self._apply_style()
        self._short_jp = SpeakerTextAccumulator()
        self._short_en = SpeakerTextAccumulator()
        self._speaker_palette = SpeakerPalette()
        self._current_source_lang = "ja"
        self._toggle_engine_fields()

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
            QTextEdit {
                background: white; border: 1px solid #d6dde6; border-radius: 6px;
                padding: 6px;
            }
            QCheckBox { color: #0b1f33; }
            """
        )

    def _start(self):
        if not self.short_toggle.isChecked():
            QtWidgets.QMessageBox.information(self, "No mode", "Enable Short.")
            return
        source_lang = "ja" if self.lang_select.currentIndex() == 0 else "en"
        engine = "whisper" if self.engine_select.currentIndex() == 0 else "lfm2"
        self._current_source_lang = source_lang
        cfg = RunConfig(
            use_loopback=self.system_audio.isChecked(),
            source_lang=source_lang,
            chunk_seconds=self.chunk_spin.value(),
            engine=engine,
            lfm2_repo=self.lfm2_repo.text().strip(),
            enable_short=self.short_toggle.isChecked(),
            enable_translation=self.translate_toggle.isChecked(),
        )
        self._engine.set_short_enabled(cfg.enable_short)
        self._engine.set_translation_enabled(cfg.enable_translation)
        self._engine.start(cfg)
        self._engine.set_source_lang(source_lang)

    def _stop(self):
        self._engine.stop()

    def _set_source_lang(self, idx: int):
        source_lang = "ja" if idx == 0 else "en"
        self._current_source_lang = source_lang
        self._engine.set_source_lang(source_lang)

    def _reset_text(self):
        self._short_jp = SpeakerTextAccumulator()
        self._short_en = SpeakerTextAccumulator()
        self._speaker_palette = SpeakerPalette()
        self.jp_text.clear()
        self.en_text.clear()

    def _save_script(self):
        if self._current_source_lang == "ja":
            short_source = self._short_jp.render_plain()
            short_trans = self._short_en.render_plain()
            src_label = "ja"
            tgt_label = "en"
        else:
            short_source = self._short_en.render_plain()
            short_trans = self._short_jp.render_plain()
            src_label = "en"
            tgt_label = "ja"

        if not (short_source.strip() or short_trans.strip()):
            QtWidgets.QMessageBox.information(
                self, "No text", "Speech recognition and translation are empty."
            )
            return

        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder to save scripts", ""
        )
        if not out_dir:
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = [
            (Path(out_dir) / f"{stamp}_short_source_{src_label}.txt", short_source),
            (Path(out_dir) / f"{stamp}_short_translation_{tgt_label}.txt", short_trans),
        ]

        try:
            for path, text in files:
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))

    @QtCore.Slot(str, str, object)
    def _append_short(self, jp: str, en: str, speaker: Optional[str]):
        if jp:
            self._short_jp.append(jp, speaker)
            self.jp_text.setHtml(self._short_jp.render_html(self._speaker_palette))
            self.jp_text.moveCursor(QtGui.QTextCursor.End)
        if en:
            self._short_en.append(en, speaker)
            self.en_text.setHtml(self._short_en.render_html(self._speaker_palette))
            self.en_text.moveCursor(QtGui.QTextCursor.End)

    @QtCore.Slot(str)
    def _set_status(self, status: str):
        self.status_lbl.setText(status)
        if status == "Listening":
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress.setVisible(False)
            self.engine_select.setEnabled(False)
            self.lfm2_repo.setEnabled(False)
        elif status in ("Idle", "Stopping..."):
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress.setVisible(False)
            self.system_audio.setEnabled(True)
            self.lang_select.setEnabled(True)
            self.chunk_spin.setEnabled(True)
            self.engine_select.setEnabled(True)
            self._toggle_engine_fields()
        elif status == "Starting...":
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress.setVisible(True)
            self.engine_select.setEnabled(False)
            self.lfm2_repo.setEnabled(False)

    @QtCore.Slot(str)
    def _show_error(self, message: str):
        QtWidgets.QMessageBox.critical(self, "Error", message)

    def _toggle_engine_fields(self):
        use_lfm2 = self.engine_select.currentIndex() == 1
        self.lfm2_repo.setEnabled(use_lfm2)


def _resource_path(rel_path: Path) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return (base / rel_path).resolve()
