import argparse
import os
import threading
from queue import Queue, Empty

import numpy as np
from transformers.utils import logging as hf_logging

from audio_io import (
    list_devices,
    load_audio,
    normalize_device_name,
    record_audio,
)
from pipeline import SpeechToEnglishPipeline
from pipeline_impl import JaToEnTranslator
from settings import DEFAULT_MIC_DEVICE, DEFAULT_MODEL_ID, LFM2_AUDIO_REPO
from lfm2_transcriber import LFM2AudioTranscriber
from whisper_transcriber import (
    WhisperOVTranscriber,
    transcribe_audio,
)

# Reduce noisy Hugging Face/Transformers warnings for this CLI.
hf_logging.set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLI for speech recognition (Whisper/OpenVINO or LFM2.5-Audio GGUF).",
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
        "--mic-device",
        default=DEFAULT_MIC_DEVICE,
        help="Input device name or index for --mic.",
    )
    parser.add_argument(
        "--output-device",
        default=None,
        help="Output device name or index for --loopback/--system-audio.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices and exit.",
    )
    parser.add_argument(
        "--loopback",
        action="store_true",
        help="Use WASAPI loopback to capture system audio.",
    )
    parser.add_argument(
        "--system-audio",
        action="store_true",
        help="Convenience flag: capture system audio (same as --loopback --mic).",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id or local path.",
    )
    parser.add_argument(
        "--engine",
        choices=["whisper", "lfm2"],
        default="whisper",
        help="Speech recognition engine.",
    )
    parser.add_argument(
        "--lfm2-repo",
        default=LFM2_AUDIO_REPO,
        help="Hugging Face repo id for liquid-audio model.",
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
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate recognized Japanese to English.",
    )
    parser.set_defaults(denoise=True)
    return parser.parse_args()


def _apply_preset(args):
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


def _run_mic(args, capture_device):
    transcriber = _build_transcriber(args)
    pipeline = (
        SpeechToEnglishPipeline(transcriber, JaToEnTranslator())
        if args.translate
        else None
    )
    print("Ready")
    stop_event = threading.Event()
    work_q: Queue[tuple[np.ndarray, int]] = Queue()

    def producer():
        while not stop_event.is_set():
            audio, sr = record_audio(
                duration_s=args.record_chunk,
                target_sr=transcriber.target_sr,
                device=capture_device,
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
            text = (
                pipeline.run(audio, sr)
                if pipeline
                else transcriber.transcribe_array(audio, sr)
            )
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


def _run_file(args):
    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    if args.translate:
        transcriber = _build_transcriber(args)
        pipeline = SpeechToEnglishPipeline(transcriber, JaToEnTranslator())
        audio, sr = load_audio(args.audio, target_sr=transcriber.target_sr)
        text = pipeline.run(audio, sr)
    else:
        if args.engine == "whisper":
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
        else:
            transcriber = _build_transcriber(args)
            audio, sr = load_audio(args.audio, target_sr=transcriber.target_sr)
            text = transcriber.transcribe_array(audio, sr)
    print(text)


def main():
    args = parse_args()
    if args.list_devices:
        list_devices()
        return
    if args.system_audio:
        args.mic = True
        args.loopback = True
    if args.mic and args.audio:
        raise ValueError("Use either an audio file or --mic, not both.")
    if not args.mic and not args.audio:
        raise ValueError("Provide an audio file or use --mic.")
    if args.loopback and not args.mic:
        args.mic = True

    if args.engine == "whisper":
        _apply_preset(args)

    mic_device = normalize_device_name(args.mic_device)
    output_device = normalize_device_name(args.output_device)
    capture_device = output_device if args.loopback else mic_device

    if args.mic:
        _run_mic(args, capture_device)
    else:
        _run_file(args)


def _build_transcriber(args):
    if args.engine == "lfm2":
        return LFM2AudioTranscriber(
            repo_id=args.lfm2_repo,
            denoise=args.denoise,
        )
    return WhisperOVTranscriber(
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


if __name__ == "__main__":
    main()
