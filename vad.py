import os

import librosa
import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from settings import (
    PROJECT_MODELS_DIR,
    VAD_MIN_SILENCE_MS,
    VAD_MIN_SPEECH_MS,
    VAD_THRESHOLD,
    VAD_WINDOW_SAMPLES,
)


class SileroVAD:
    def __init__(
        self,
        device: str | None = None,
        threshold: float = VAD_THRESHOLD,
        min_speech_ms: int = VAD_MIN_SPEECH_MS,
        min_silence_ms: int = VAD_MIN_SILENCE_MS,
        window_size_samples: int = VAD_WINDOW_SAMPLES,
    ):
        os.environ.setdefault("TORCH_HOME", PROJECT_MODELS_DIR)
        self._device = torch.device(device or "cpu")
        self._threshold = float(threshold)
        self._min_speech_ms = int(min_speech_ms)
        self._min_silence_ms = int(min_silence_ms)
        self._window_size_samples = int(window_size_samples)
        self._model = load_silero_vad()
        self._model = self._model.to(self._device)
        self._model.eval()

    def has_speech(self, audio: np.ndarray, sr: int) -> bool:
        if audio is None or audio.size == 0:
            return False
        audio = np.asarray(audio, dtype=np.float32)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        if audio.size == 0:
            return False
        tensor = torch.from_numpy(audio).to(self._device)
        with torch.no_grad():
            speech = get_speech_timestamps(
                tensor,
                self._model,
                sampling_rate=sr,
                threshold=self._threshold,
                min_speech_duration_ms=self._min_speech_ms,
                min_silence_duration_ms=self._min_silence_ms,
                window_size_samples=self._window_size_samples,
            )
        return bool(speech)
