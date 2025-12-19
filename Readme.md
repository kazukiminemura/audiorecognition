# Audio Recognition CLI (OpenVINO whisper-large-v3-int4-ov)

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python transcribe.py path\to\audio.wav
```

Common options:

```bash
python transcribe.py audio.mp3 --language ja --task transcribe --device CPU
```

Record from microphone:

```bash
python transcribe.py --mic --duration 5
```

Capture system audio with WASAPI loopback:

```bash
python transcribe.py --mic --loopback
```

List devices to find a WASAPI output device:

```bash
python transcribe.py --list-devices
```

## Notes

- The first run downloads the model from Hugging Face.
- For long audio, the app uses chunking with overlap. Use `--chunk-length 0` to disable.
- If mp3 loading fails, install ffmpeg and retry.
- If microphone recording fails, ensure the input device is available and set `--mic-device`.
