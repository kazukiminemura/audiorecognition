import librosa
import numpy as np
import soundfile as sf
import pyaudiowpatch as pyaudio

from settings import (
    DEFAULT_MIC_DEVICE,
    FRAMES_PER_BUFFER,
    MIC_INPUT_SAMPLE_RATE,
    SAMPLE_FORMAT,
)


def load_audio(path, target_sr):
    audio, sr = sf.read(path)
    if audio.size == 0:
        return audio, sr
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_sr is not None and sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr


def _list_devices(pa):
    print("Available devices:")
    for idx in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(idx)
        flags = []
        if info.get("maxInputChannels", 0) > 0:
            flags.append("IN")
        if info.get("maxOutputChannels", 0) > 0:
            flags.append("OUT")
        if info.get("isLoopbackDevice"):
            flags.append("LOOP")
        flags_text = ",".join(flags) if flags else "N/A"
        print(f"[{idx}] {info['name']} ({flags_text})")


def list_devices():
    pa = pyaudio.PyAudio()
    try:
        _list_devices(pa)
    finally:
        pa.terminate()


def _get_default_output(pa):
    api = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    return pa.get_device_info_by_index(api["defaultOutputDevice"])


def _get_default_input(pa):
    api = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    return pa.get_device_info_by_index(api["defaultInputDevice"])


def _is_loopback_device(info):
    return bool(info.get("isLoopbackDevice"))


def _find_device_by_name(
    pa,
    name_substring,
    require_input=False,
    require_output=False,
    require_loopback=False,
):
    needle = name_substring.lower()
    for idx in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(idx)
        if needle not in info["name"].lower():
            continue
        if require_input and info.get("maxInputChannels", 0) <= 0:
            continue
        if require_output and info.get("maxOutputChannels", 0) <= 0:
            continue
        if require_loopback and not _is_loopback_device(info):
            continue
        return info
    return None


def _find_input_device(pa, device):
    if device is None:
        return _get_default_input(pa)
    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        info = pa.get_device_info_by_index(int(device))
        if info.get("maxInputChannels", 0) <= 0:
            raise RuntimeError("Selected device is not an input device.")
        return info
    info = _find_device_by_name(pa, str(device), require_input=True)
    if info is None:
        _list_devices(pa)
        raise RuntimeError("No matching input device found.")
    return info


def _find_loopback_for_output(pa, output_info):
    for loop_info in pa.get_loopback_device_info_generator():
        if output_info["name"] in loop_info["name"]:
            return loop_info
    raise RuntimeError("Loopback device for selected output not found.")


def _find_loopback_device(pa, device):
    if device is None:
        out_info = _get_default_output(pa)
        return _find_loopback_for_output(pa, out_info)

    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        info = pa.get_device_info_by_index(int(device))
        if _is_loopback_device(info):
            return info
        if info.get("maxOutputChannels", 0) > 0:
            return _find_loopback_for_output(pa, info)
        raise RuntimeError("Selected device is not an output/loopback device.")

    device_str = str(device)
    loop_info = _find_device_by_name(
        pa, device_str, require_input=True, require_loopback=True
    )
    if loop_info is not None:
        return loop_info
    out_info = _find_device_by_name(pa, device_str, require_output=True)
    if out_info is not None:
        return _find_loopback_for_output(pa, out_info)
    _list_devices(pa)
    raise RuntimeError("No matching output/loopback device found.")


def _read_stream_frames(stream, total_frames):
    chunks = []
    recorded_frames = 0
    while recorded_frames < total_frames:
        frames = min(FRAMES_PER_BUFFER, total_frames - recorded_frames)
        data = stream.read(frames, exception_on_overflow=False)
        chunks.append(data)
        recorded_frames += frames
    return b"".join(chunks)


def _bytes_to_float32(data_bytes, channels):
    if not data_bytes:
        return np.zeros((0, channels), dtype="float32")
    data_i16 = np.frombuffer(data_bytes, dtype=np.int16)
    if channels > 1:
        data_i16 = data_i16.reshape((-1, channels))
    return (data_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def record_audio(duration_s, target_sr, device=None, loopback=False):
    pa = pyaudio.PyAudio()
    try:
        if loopback:
            dev_info = _find_loopback_device(pa, device)
            max_ch = int(dev_info.get("maxInputChannels", 2))
            channels = 2 if max_ch >= 2 else 1
        else:
            dev_info = _find_input_device(pa, device)
            max_ch = int(dev_info.get("maxInputChannels", 1))
            channels = 1 if max_ch >= 1 else max_ch

        if channels <= 0:
            raise RuntimeError("Selected device does not expose any usable channels.")

        samplerate = (
            int(dev_info.get("defaultSampleRate", MIC_INPUT_SAMPLE_RATE))
            if loopback
            else MIC_INPUT_SAMPLE_RATE
        )
        stream = pa.open(
            format=SAMPLE_FORMAT,
            channels=channels,
            rate=samplerate,
            input=True,
            input_device_index=dev_info["index"],
            frames_per_buffer=FRAMES_PER_BUFFER,
        )

        try:
            total_frames = int(duration_s * samplerate)
            raw = _read_stream_frames(stream, total_frames)
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
    finally:
        pa.terminate()

    audio = _bytes_to_float32(raw, channels)
    if audio.size == 0:
        return audio, samplerate
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    input_sr = samplerate
    if target_sr is not None and input_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=input_sr, target_sr=target_sr)
        input_sr = target_sr
    return audio, input_sr


def normalize_device_name(device):
    if device is None:
        return None
    device_str = str(device).strip().lower()
    if device_str in ("", "default", "system"):
        return None
    return device


def default_mic_device():
    return DEFAULT_MIC_DEVICE
