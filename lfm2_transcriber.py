import os

import numpy as np
import torch
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

from audio_processing import denoise_audio
from settings import (
    FIXED_SAMPLE_RATE,
    LFM2_AUDIO_DEVICE,
    LFM2_AUDIO_DTYPE,
    LFM2_AUDIO_MAX_NEW_TOKENS,
    LFM2_AUDIO_REPO,
    LFM2_AUDIO_SYSTEM_PROMPT,
    LFM2_AUDIO_HF_HOME,
    LFM2_AUDIO_DISABLE_SYMLINKS,
)


def _parse_dtype(dtype_name: str) -> torch.dtype:
    name = (dtype_name or "float32").lower()
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


class LFM2AudioTranscriber:
    """Transcriber backed by liquid-audio (Hugging Face download)."""

    def __init__(
        self,
        repo_id: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        denoise: bool = True,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
    ):
        self.target_sr = FIXED_SAMPLE_RATE
        self.denoise = denoise
        self._repo_id = repo_id or LFM2_AUDIO_REPO
        requested_device = device or LFM2_AUDIO_DEVICE
        if requested_device and str(requested_device).lower() != "cpu":
            # liquid-audio currently runs CPU-only in this project
            self._device = "cpu"
        else:
            self._device = "cpu"
        self._dtype = _parse_dtype(dtype or LFM2_AUDIO_DTYPE)
        self._system_prompt = system_prompt or LFM2_AUDIO_SYSTEM_PROMPT
        self._max_new_tokens = (
            int(max_new_tokens)
            if max_new_tokens is not None
            else int(LFM2_AUDIO_MAX_NEW_TOKENS)
        )

        if LFM2_AUDIO_DISABLE_SYMLINKS:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        if LFM2_AUDIO_HF_HOME:
            os.environ.setdefault("HF_HOME", LFM2_AUDIO_HF_HOME)

        self._processor = LFM2AudioProcessor.from_pretrained(
            self._repo_id, device=self._device
        ).eval()
        self._model = LFM2AudioModel.from_pretrained(
            self._repo_id, device=self._device, dtype=self._dtype
        ).eval()
        self._model = self._model.to(self._device)
        if self._dtype is not None:
            self._model = self._model.to(dtype=self._dtype)
        self._model.eval()

    def transcribe_array(self, audio, sr) -> str:
        if audio is None or len(audio) == 0:
            return ""
        audio = np.asarray(audio, dtype=np.float32)
        if sr != self.target_sr:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        if self.denoise:
            audio = denoise_audio(audio, sr)
        audio = np.clip(audio, -1.0, 1.0)

        wav = torch.from_numpy(audio).unsqueeze(0)
        if self._device:
            wav = wav.to(self._device)

        chat = ChatState(self._processor, dtype=self._dtype)
        if self._system_prompt:
            chat.new_turn("system")
            chat.add_text(self._system_prompt)
            chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(wav, sr)
        chat.end_turn()
        chat.new_turn("assistant")

        text_token_ids: list[int] = []
        with torch.inference_mode():
            for token in self._model.generate_sequential(
                **chat, max_new_tokens=self._max_new_tokens
            ):
                if token.numel() == 1:
                    text_token_ids.append(int(token.item()))

        if not text_token_ids:
            return ""
        return self._processor.text.decode(
            text_token_ids,
            skip_special_tokens=True,
        ).strip()
