import os
from pathlib import Path
import pyaudiowpatch as pyaudio

DEFAULT_MODEL_ID = "OpenVINO/whisper-large-v3-fp16-ov"
TRANSLATION_MODEL_ID = "Helsinki-NLP/opus-mt-jap-en"
TRANSLATION_MODEL_ID_EN_JA = "Helsinki-NLP/opus-mt-en-jap"
FIXED_SAMPLE_RATE = 16000
MIC_INPUT_SAMPLE_RATE = 48000
DEFAULT_MIC_DEVICE = None
SAMPLE_FORMAT = pyaudio.paInt16
FRAMES_PER_BUFFER = 1024

LFM2_AUDIO_REPO = os.getenv("LFM2_AUDIO_REPO", "LiquidAI/LFM2.5-Audio-1.5B")
LFM2_AUDIO_SYSTEM_PROMPT = os.getenv("LFM2_AUDIO_SYSTEM_PROMPT", "Perform ASR.")
LFM2_AUDIO_MAX_NEW_TOKENS = int(os.getenv("LFM2_AUDIO_MAX_NEW_TOKENS", "256"))
LFM2_AUDIO_DTYPE = os.getenv("LFM2_AUDIO_DTYPE", "float32")
LFM2_AUDIO_DEVICE = os.getenv("LFM2_AUDIO_DEVICE", "cpu")
PROJECT_MODELS_DIR = os.getenv(
    "MODELS_DIR",
    str((Path(__file__).resolve().parent / "models").resolve()),
)
LFM2_AUDIO_HF_HOME = os.getenv("LFM2_AUDIO_HF_HOME", PROJECT_MODELS_DIR)
LFM2_AUDIO_DISABLE_SYMLINKS = os.getenv("LFM2_AUDIO_DISABLE_SYMLINKS", "1") != "0"

WARNED_DENOISE = set()
