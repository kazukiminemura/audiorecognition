import os

import librosa
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor

from audio_io import load_audio, record_audio
from audio_processing import build_forced_decoder_ids, chunk_audio, denoise_audio
from settings import DEFAULT_MODEL_ID, FIXED_SAMPLE_RATE, PROJECT_MODELS_DIR


def prepare_model(model_id, device):
    os.environ.setdefault("HF_HOME", PROJECT_MODELS_DIR)
    processor = AutoProcessor.from_pretrained(model_id)
    model = OVModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        device=device,
        compile=True,
    )
    target_sr = FIXED_SAMPLE_RATE
    return processor, model, target_sr


class WhisperOVTranscriber:
    """Reusable transcriber that can be imported by other apps."""

    def __init__(
        self,
        model_id=DEFAULT_MODEL_ID,
        device="AUTO",
        language="ja",
        task="transcribe",
        chunk_length_s=30.0,
        stride_length_s=5.0,
        max_new_tokens=128,
        denoise=True,
        num_beams=1,
    ):
        self.language = language
        self.task = task
        self.chunk_length_s = chunk_length_s
        self.stride_length_s = stride_length_s
        self.max_new_tokens = max_new_tokens
        self.denoise = denoise
        self.num_beams = num_beams
        self.processor, self.model, self.target_sr = prepare_model(model_id, device)

    def transcribe_array(self, audio, sr):
        return transcribe_array(
            self.processor,
            self.model,
            audio,
            sr,
            self.language,
            self.task,
            self.chunk_length_s,
            self.stride_length_s,
            self.max_new_tokens,
            self.denoise,
            self.num_beams,
        )

    def transcribe_file(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio, sr = load_audio(audio_path, target_sr=self.target_sr)
        return self.transcribe_array(audio, sr)

    def transcribe_mic_chunk(self, duration_s=3.0, mic_device=None, loopback=False):
        """Record once from mic and return text; good for GUI button-style usage."""
        audio, sr = record_audio(
            duration_s=duration_s,
            target_sr=self.target_sr,
            device=mic_device,
            loopback=loopback,
        )
        return self.transcribe_array(audio, sr)


def transcribe_array(
    processor,
    model,
    audio,
    sr,
    language,
    task,
    chunk_length_s,
    stride_length_s,
    max_new_tokens,
    denoise,
    num_beams,
):
    target_sr = FIXED_SAMPLE_RATE
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    forced_decoder_ids = build_forced_decoder_ids(processor, language, task)

    segments = []
    for chunk in chunk_audio(audio, sr, chunk_length_s, stride_length_s):
        if denoise:
            chunk = denoise_audio(chunk, sr)
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
        generated_ids = model.generate(
            inputs.input_features,
            forced_decoder_ids=forced_decoder_ids,
            language=language if language else None,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        segments.append(text.strip())

    return " ".join([t for t in segments if t])


def transcribe_audio(
    model_id,
    audio_path,
    device,
    language,
    task,
    chunk_length_s,
    stride_length_s,
    max_new_tokens,
    denoise,
    num_beams,
):
    transcriber = WhisperOVTranscriber(
        model_id=model_id,
        device=device,
        language=language,
        task=task,
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
        max_new_tokens=max_new_tokens,
        denoise=denoise,
        num_beams=num_beams,
    )
    return transcriber.transcribe_file(audio_path)
