import argparse
import os
import threading
from queue import Queue, Empty

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor


DEFAULT_MODEL_ID = "OpenVINO/whisper-large-v3-fp16-ov"
_WARNED_SAMPLERATES = set()
_WARNED_DENOISE = set()
# Use None so sounddevice picks the OS default input device.
DEFAULT_MIC_DEVICE = None


def load_audio(path, target_sr):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_sr is not None and sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr


def list_devices():
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    for idx, dev in enumerate(devices):
        hostapi_name = hostapis[dev["hostapi"]]["name"]
        direction = []
        if dev["max_input_channels"] > 0:
            direction.append("in")
        if dev["max_output_channels"] > 0:
            direction.append("out")
        direction_str = "/".join(direction) if direction else "none"
        print(f"[{idx}] {dev['name']} ({hostapi_name}, {direction_str})")


def resolve_loopback_device(device):
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    def is_wasapi(dev):
        hostapi_name = hostapis[dev["hostapi"]]["name"]
        return "WASAPI" in hostapi_name.upper()

    def is_output(dev):
        return dev["max_output_channels"] > 0

    if device is None:
        default_out = sd.default.device[1]
        if default_out is not None:
            dev = devices[default_out]
            if is_wasapi(dev) and is_output(dev):
                return default_out
        for idx, dev in enumerate(devices):
            if is_wasapi(dev) and is_output(dev):
                return idx
        raise RuntimeError("No WASAPI output device found for loopback.")

    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        idx = int(device)
        dev = devices[idx]
        if not is_wasapi(dev):
            raise RuntimeError("Selected device is not WASAPI. Use a WASAPI output device.")
        if not is_output(dev):
            raise RuntimeError("Selected device is not an output device for loopback.")
        return idx

    needle = str(device).lower()
    for idx, dev in enumerate(devices):
        if needle in dev["name"].lower() and is_wasapi(dev) and is_output(dev):
            return idx
    raise RuntimeError("No matching WASAPI output device for loopback.")


def pick_input_samplerate(target_sr, device, channels, extra_settings):
    if target_sr is None:
        device_info = sd.query_devices(device)
        return int(device_info["default_samplerate"])
    try:
        sd.check_input_settings(
            device=device,
            channels=channels,
            samplerate=target_sr,
            extra_settings=extra_settings,
        )
        return target_sr
    except Exception:
        device_info = sd.query_devices(device)
        fallback_sr = int(device_info["default_samplerate"])
        key = (
            device_info["name"],
            device_info["hostapi"],
            channels,
            target_sr,
            bool(extra_settings),
        )
        if key not in _WARNED_SAMPLERATES:
            print(
                f"Requested samplerate {target_sr} not supported; using {fallback_sr}."
            )
            _WARNED_SAMPLERATES.add(key)
        return fallback_sr


def record_audio(duration_s, target_sr, device=None, loopback=False):
    extra_settings = None
    channels = 1
    if loopback:
        if not hasattr(sd, "WasapiSettings"):
            raise RuntimeError("WASAPI loopback is only supported on Windows.")
        extra_settings = sd.WasapiSettings(True)
        device = resolve_loopback_device(device)
        channels = 2

    input_sr = pick_input_samplerate(target_sr, device, channels, extra_settings)
    frames = int(duration_s * input_sr)
    recording = sd.rec(
        frames,
        samplerate=input_sr,
        channels=channels,
        dtype="float32",
        device=device,
        extra_settings=extra_settings,
    )
    sd.wait()
    audio = recording.squeeze()
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_sr is not None and input_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=input_sr, target_sr=target_sr)
        input_sr = target_sr
    return audio, input_sr


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
        if key not in _WARNED_DENOISE:
            print(f"Noise reduction failed; using raw audio. ({exc})")
            _WARNED_DENOISE.add(key)
        return audio


def build_forced_decoder_ids(processor, language, task):
    if language or task:
        return processor.get_decoder_prompt_ids(
            language=language if language else None,
            task=task if task else None,
        )
    return None


def prepare_model(model_id, device):
    processor = AutoProcessor.from_pretrained(model_id)
    model = OVModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        device=device,
        compile=True,
    )
    target_sr = processor.feature_extractor.sampling_rate
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLI for speech recognition with OpenVINO Whisper models.",
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="Path to an audio file (wav/mp3/flac).",
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Record from microphone instead of loading a file.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds for --mic.",
    )
    parser.add_argument(
        "--mic-device",
        default=DEFAULT_MIC_DEVICE,
        help="Input device name or index for --mic. For --loopback, use an output device.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices and exit.",
    )
    parser.add_argument(
        "--loopback",
        action="store_true",
        help="Use WASAPI loopback to capture system audio (requires --mic).",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id or local path.",
    )
    parser.add_argument(
        "--device",
        default="GPU",
        help="OpenVINO device (CPU, GPU, AUTO, etc.).",
    )
    parser.add_argument("--language", default="ja", help="Language code, e.g. ja, en.")
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Whisper task hint.",
    )
    parser.add_argument(
        "--chunk-length",
        type=float,
        default=30.0,
        help="Chunk length in seconds. Use 0 to disable chunking.",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=5.0,
        help="Stride overlap in seconds for chunking.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Generation length cap.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam search width (1 = greedy, faster).",
    )
    parser.add_argument(
        "--no-denoise",
        dest="denoise",
        action="store_false",
        help="Disable pre-processing noise reduction.",
    )
    parser.add_argument(
        "--preset",
        choices=["balanced", "fast", "quality"],
        default="quality",
        help="Tuning shortcuts: fast (tiny int8, shorter chunks), balanced (small fp16), quality (large fp16, higher beams).",
    )
    parser.add_argument(
        "--record-chunk",
        type=float,
        default=4.0,
        help="Per-chunk recording duration in seconds when using --mic.",
    )
    parser.set_defaults(denoise=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_devices:
        list_devices()
        return
    if args.mic and args.audio:
        raise ValueError("Use either an audio file or --mic, not both.")
    if not args.mic and not args.audio:
        raise ValueError("Provide an audio file or use --mic.")
    if args.mic and args.duration <= 0:
        raise ValueError("--duration must be > 0.")
    if args.loopback and not args.mic:
        raise ValueError("--loopback requires --mic.")

    # Apply speed/quality presets (override conflicting params).
    if args.preset == "fast":
        args.model_id = "OpenVINO/whisper-tiny-int8-ov"
        args.chunk_length = 10.0
        args.stride = 2.0
        args.max_new_tokens = 64
        args.num_beams = 1
    elif args.preset == "balanced":
        args.model_id = "OpenVINO/whisper-small-fp16-ov"
        args.chunk_length = 15.0
        args.stride = 3.0
        args.max_new_tokens = 96
        args.num_beams = 1
    elif args.preset == "quality":
        args.model_id = "OpenVINO/whisper-large-v3-fp16-ov"
        args.chunk_length = 35.0
        args.stride = 7.0
        args.max_new_tokens = 160
        args.num_beams = 4

    # Normalize mic device: None / "" / "default" -> OS default input.
    mic_device = (
        None
        if args.mic_device is None
        or str(args.mic_device).strip().lower() in ("", "default", "system")
        else args.mic_device
    )

    if args.mic:
        transcriber = WhisperOVTranscriber(
            model_id=args.model_id,
            device=args.device,
            language=args.language,
            task=args.task,
            chunk_length_s=args.chunk_length,
            stride_length_s=args.stride,
            max_new_tokens=args.max_new_tokens,
            denoise=args.denoise,
            num_beams=args.num_beams,
        )
        print("Ready")
        stop_event = threading.Event()
        work_q: Queue[tuple[np.ndarray, int]] = Queue()

        def producer():
            while not stop_event.is_set():
                audio, sr = record_audio(
                    duration_s=args.record_chunk,
                    target_sr=transcriber.target_sr,
                    device=mic_device,
                    loopback=args.loopback,
                )
                # Block instead of dropping to guarantee no chunks are skipped.
                work_q.put((audio, sr))

        def consumer():
            while not stop_event.is_set() or not work_q.empty():
                try:
                    audio, sr = work_q.get(timeout=0.5)
                except Empty:
                    continue
                text = transcriber.transcribe_array(audio, sr)
                if text:
                    print(text)
                work_q.task_done()

        prod_t = threading.Thread(target=producer, daemon=True)
        cons_t = threading.Thread(target=consumer, daemon=True)
        prod_t.start()
        cons_t.start()
        try:
            while prod_t.is_alive() and cons_t.is_alive():
                prod_t.join(timeout=0.5)
                cons_t.join(timeout=0.5)
        except KeyboardInterrupt:
            stop_event.set()
            prod_t.join()
            cons_t.join()
            return
    else:
        if not os.path.exists(args.audio):
            raise FileNotFoundError(f"Audio file not found: {args.audio}")
        text = transcribe_audio(
            model_id=args.model_id,
            audio_path=args.audio,
            device=args.device,
            language=args.language,
            task=args.task,
            chunk_length_s=args.chunk_length,
            stride_length_s=args.stride,
            max_new_tokens=args.max_new_tokens,
            denoise=args.denoise,
            num_beams=args.num_beams,
        )
    print(text)


if __name__ == "__main__":
    main()

