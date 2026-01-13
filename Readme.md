# Audio Recognition (Speech, Translation, Minutes)

Desktop UI + CLI for speech recognition with Whisper/OpenVINO or LiquidAI LFM2.5-Audio,
plus Japanese ⇄ English translation and meeting minutes generation.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## UI (recommended)

```bash
python ui.py
```

The UI can:
- Record microphone audio or capture system audio (WASAPI loopback).
- Transcribe with Whisper (OpenVINO) or LFM2.5-Audio (CPU).
- Translate between Japanese and English.
- Generate minutes automatically every 3 minutes or from a saved script.

Files are saved to `outputs\minutes` (script + minutes).

## CLI

```bash
python transcribe.py path\to\audio.wav
```

Common options:

```bash
python transcribe.py audio.mp3 --device CPU --language ja --task transcribe
python transcribe.py --engine lfm2 --lfm2-repo LiquidAI/LFM2.5-Audio-1.5B
python transcribe.py --mic --record-chunk 4
python transcribe.py --system-audio
python transcribe.py --list-devices
python transcribe.py --translate
python transcribe.py --preset fast
```

## Model download (optional)

```bash
python download_models.py
python download_models.py --core-only
```

## Notes

- The first run downloads models from Hugging Face (UI auto-preloads unless disabled).
- For long audio, the CLI uses chunking with overlap. Use `--chunk-length 0` to disable.
- If mp3 loading fails, install ffmpeg and retry.
- If microphone recording fails, ensure the input device is available and set `--mic-device`.
