import time
import numpy as np
import soundfile as sf
import pyaudiowpatch as pyaudio

output_file = "system_audio.wav"
record_sec = 10
pause = False
is_silence_cut = False
silence_threshold = 0.01
# Mode: "loopback" records speaker output, "microphone" records mic input.
record_mode = "loopback"
target_speaker_name = None  # e.g. "Realtek" or "HDMI"
target_mic_name = None  # e.g. "USB Mic"
target_samplerate = 48000  # e.g. 48000 or 44100
target_channels = 2  # e.g. 1 or 2
frames_per_buffer = 1024
sample_format = pyaudio.paInt16


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
        print(f"  - [{idx}] {info['name']} ({flags_text})")


def _get_default_output(pa):
    api = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    return pa.get_device_info_by_index(api["defaultOutputDevice"])


def _find_output_device(pa, name_substring=None):
    if name_substring:
        name_substring = name_substring.lower()
        for idx in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(idx)
            if info.get("maxOutputChannels", 0) <= 0:
                continue
            if name_substring in info["name"].lower():
                return info
        print(f"Speaker matching '{name_substring}' not found.")
        _list_devices(pa)
        raise SystemExit(1)
    return _get_default_output(pa)


def _find_loopback_for_output(pa, output_info):
    for loop_info in pa.get_loopback_device_info_generator():
        if output_info["name"] in loop_info["name"]:
            return loop_info
    print("Loopback device for default output not found.")
    _list_devices(pa)
    raise SystemExit(1)


def _find_input_device(pa, name_substring=None):
    if name_substring:
        name_substring = name_substring.lower()
        for idx in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(idx)
            if info.get("maxInputChannels", 0) <= 0:
                continue
            if name_substring in info["name"].lower():
                return info
        print(f"Microphone matching '{name_substring}' not found.")
        _list_devices(pa)
        raise SystemExit(1)
    api = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    return pa.get_device_info_by_index(api["defaultInputDevice"])


def _read_stream_frames(stream, total_frames, channels, samplerate):
    chunks = []
    recorded_frames = 0
    while recorded_frames < total_frames:
        if pause:
            time.sleep(0.05)
            continue

        frames = min(frames_per_buffer, total_frames - recorded_frames)
        data = stream.read(frames, exception_on_overflow=False)
        chunks.append(data)
        recorded_frames += frames

        # Peak meter (int16)
        np_chunk = np.frombuffer(data, dtype=np.int16)
        peak = float(np.max(np.abs(np_chunk))) / 32768.0 if np_chunk.size else 0.0
        recorded_sec = recorded_frames / samplerate
        print(f"\rREC {recorded_sec:6.1f}s / {record_sec}s  peak={peak:.5f}", end="")
    return b"".join(chunks)


def _bytes_to_float32(data_bytes, channels):
    if not data_bytes:
        return np.zeros((0, channels), dtype="float32")
    data_i16 = np.frombuffer(data_bytes, dtype=np.int16)
    if channels > 1:
        data_i16 = data_i16.reshape((-1, channels))
    return (data_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def main():
    pa = pyaudio.PyAudio()
    try:
        if record_mode == "microphone":
            mic_info = _find_input_device(pa, target_mic_name)
            samplerate = int(target_samplerate or mic_info.get("defaultSampleRate", 44100))
            max_ch = int(mic_info.get("maxInputChannels", 1))
            channels = int(target_channels or max_ch)
            channels = min(channels, max_ch)

            print(
                f"Recording microphone '{mic_info['name']}' for {record_sec}s "
                f"at {samplerate} Hz, channels={channels}..."
            )

            stream = pa.open(
                format=sample_format,
                channels=channels,
                rate=samplerate,
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=frames_per_buffer,
            )
        else:
            out_info = _find_output_device(pa, target_speaker_name)
            loop_info = _find_loopback_for_output(pa, out_info)
            samplerate = int(target_samplerate or loop_info.get("defaultSampleRate", 48000))
            max_ch = int(loop_info.get("maxInputChannels", 2))
            channels = int(target_channels or max_ch)
            channels = min(channels, max_ch)

            print(
                f"Recording speaker '{out_info['name']}' via loopback for {record_sec}s "
                f"at {samplerate} Hz, channels={channels}..."
            )

            stream = pa.open(
                format=sample_format,
                channels=channels,
                rate=samplerate,
                input=True,
                input_device_index=loop_info["index"],
                frames_per_buffer=frames_per_buffer,
            )

        total_frames = int(record_sec * samplerate)
        try:
            raw = _read_stream_frames(stream, total_frames, channels, samplerate)
        finally:
            stream.stop_stream()
            stream.close()
    finally:
        pa.terminate()

    data = _bytes_to_float32(raw, channels)
    if is_silence_cut:
        mask = np.any(np.abs(data) >= silence_threshold, axis=1) if data.size else np.array([], dtype=bool)
        data = data[mask]
    sf.write(output_file, data, samplerate, subtype="PCM_16")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
