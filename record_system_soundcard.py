import sounddevice as sd
import soundfile as sf

output_file_name = "out.wav"
samplerate = 44100
record_sec = 5
stereo_mix_device = None  # set device index for "ステレオ ミキサー" if needed


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


def main():
    device = stereo_mix_device
    if device is None:
        # Try to auto-detect Stereo Mix
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            name = dev["name"]
            if dev["max_input_channels"] > 0 and ("Stereo Mix" in name or "ステレオ" in name):
                device = idx
                break

    if device is None:
        print("Stereo Mix device not found. Set stereo_mix_device or check --list-devices.")
        list_devices()
        return

    audio = sd.rec(
        int(record_sec * samplerate),
        samplerate=samplerate,
        channels=2,
        dtype="float32",
        device=device,
    )
    sd.wait()
    sf.write(output_file_name, audio, samplerate)
    print(f"Saved: {output_file_name}")


if __name__ == "__main__":
    main()
