# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

root = Path(os.environ.get("AUDIOREC_ROOT", Path.cwd())).resolve()
entrypoint = root / "ui.py"

datas = []
binaries = []
hiddenimports = []

for pkg in [
    "torch",
    "transformers",
    "sentencepiece",
    "openvino",
    "optimum",
    "optimum.intel",
    "soundfile",
    "librosa",
    "pyaudiowpatch",
    "numpy",
    "PySide6",
]:
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

a = Analysis(
    [str(entrypoint)],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="audiorecognition",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="audiorecognition",
)
