import numpy as np
import librosa

from settings import WARNED_DENOISE


def chunk_audio(audio, sr, chunk_length_s, stride_length_s):
    if chunk_length_s <= 0:
        yield audio
        return

    chunk_len = int(chunk_length_s * sr)
    stride = int(stride_length_s * sr)
    if chunk_len <= 0 or chunk_len <= stride:
        yield audio
        return

    step = chunk_len - stride
    total = len(audio)
    start = 0
    while start < total:
        end = min(start + chunk_len, total)
        yield audio[start:end]
        if end >= total:
            break
        start += step


def denoise_audio(audio, sr, noise_seconds=0.5):
    """Lightweight spectral subtraction using the first part of the clip as noise profile."""
    if sr <= 0 or audio.size == 0:
        return audio

    noise_len = min(len(audio), int(noise_seconds * sr))
    # Skip if clip is too short to build a profile.
    if noise_len < max(int(0.05 * sr), 32):
        return audio

    try:
        n_fft = 2048
        hop = 512
        noise_clip = audio[:noise_len]
        noise_stft = librosa.stft(noise_clip, n_fft=n_fft, hop_length=hop)
        noise_mag = np.abs(noise_stft).mean(axis=1, keepdims=True)

        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
        mag, phase = np.abs(stft), np.angle(stft)
        mag_denoised = np.maximum(mag - noise_mag, 0)

        cleaned = librosa.istft(
            mag_denoised * np.exp(1j * phase), hop_length=hop, length=len(audio)
        )
        return cleaned.astype(np.float32)
    except Exception as exc:
        key = type(exc).__name__
        if key not in WARNED_DENOISE:
            print(f"Noise reduction failed; using raw audio. ({exc})")
            WARNED_DENOISE.add(key)
        return audio


def build_forced_decoder_ids(processor, language, task):
    if language or task:
        return processor.get_decoder_prompt_ids(
            language=language if language else None,
            task=task if task else None,
        )
    return None
