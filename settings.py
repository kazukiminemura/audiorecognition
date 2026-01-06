import pyaudiowpatch as pyaudio

DEFAULT_MODEL_ID = "OpenVINO/whisper-large-v3-fp16-ov"
TRANSLATION_MODEL_ID = "Helsinki-NLP/opus-mt-ja-en"
FIXED_SAMPLE_RATE = 16000
MIC_INPUT_SAMPLE_RATE = 48000
DEFAULT_MIC_DEVICE = None
SAMPLE_FORMAT = pyaudio.paInt16
FRAMES_PER_BUFFER = 1024

WARNED_DENOISE = set()
