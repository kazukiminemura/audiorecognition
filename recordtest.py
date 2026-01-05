import sounddevice as sd
import soundfile as sf

target_name = "FFF-LD27P6"
output_file = "system_audio.wav"
record_sec = 10
channels = 2


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


def find_wasapi_output_device(name_substring):
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    for idx, dev in enumerate(devices):
        if dev["max_output_channels"] <= 0:
            continue
        if name_substring not in dev["name"]:
            continue
        hostapi_name = hostapis[dev["hostapi"]]["name"]
        if hostapi_name == "Windows WASAPI":
            return idx, dev
    return None, None


def main():
    device_index, device_info = find_wasapi_output_device(target_name)
    if device_index is None:
        print(f"WASAPI output device '{target_name}' not found.")
        print("Tip: pick the WASAPI variant of the device from the list below.")
        list_devices()
        return

    samplerate = int(device_info["default_samplerate"])
    rec_channels = min(channels, device_info["max_output_channels"])

    extra = sd.WasapiSettings(loopback=True)

    print(f"Recording '{device_info['name']}' for {record_sec}s at {samplerate} Hz...")
    audio = sd.rec(
        int(record_sec * samplerate),
        samplerate=samplerate,
        channels=rec_channels,
        dtype="float32",
        device=device_index,
        extra_settings=extra,
    )
    sd.wait()

    sf.write(output_file, audio, samplerate)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
