import os
from pathlib import Path
import pyaudiowpatch as pyaudio

DEFAULT_MODEL_ID = "OpenVINO/whisper-large-v3-fp16-ov"
TRANSLATION_MODEL_ID = "facebook/nllb-200-distilled-600M"
TRANSLATION_MODEL_ID_EN_JA = "facebook/nllb-200-distilled-600M"
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

VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "250"))
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "100"))
VAD_WINDOW_SAMPLES = int(os.getenv("VAD_WINDOW_SAMPLES", "512"))

MINUTES_MODEL_ID = os.getenv("MINUTES_MODEL_ID", "LiquidAI/LFM2.5-1.2B-Instruct")
MINUTES_MAX_NEW_TOKENS = int(os.getenv("MINUTES_MAX_NEW_TOKENS", "256"))

WARNED_DENOISE = set()
