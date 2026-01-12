from src import hf_env  # ensure HF env defaults before other imports
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from optimum.intel import OVModelForCausalLM
try:
    from optimum.intel.openvino.modeling_visual_language import MODEL_TYPE_TO_CLS_MAPPING
except Exception:
    MODEL_TYPE_TO_CLS_MAPPING = {}

from src.settings import (
    MINUTES_BACKEND,
    MINUTES_DEVICE,
    MINUTES_MAX_INPUT_CHARS,
    MINUTES_MAX_NEW_TOKENS,
    MINUTES_MODEL_ID,
    PROJECT_MODELS_DIR,
)

_SUMMARIZER = None


class MinutesSummarizer:
    def __init__(self, model_id: str = MINUTES_MODEL_ID):
        os.environ.setdefault("HF_HOME", PROJECT_MODELS_DIR)
        # Ensure defaults are set even if env changes after import.
        hf_env._init_hf_env()
        self._model_id = model_id
        print(f"[minutes] init tokenizer model_id={model_id}", flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        backend = (MINUTES_BACKEND or "openvino").lower()
        if backend == "openvino":
            device = (MINUTES_DEVICE or "CPU")
            device = device.strip()
            if len(device) >= 2 and device[0] == device[-1] == '"':
                device = device[1:-1].strip()
            print(f"[minutes] openvino device={device}", flush=True)
            print("[minutes] loading openvino model...", flush=True)
            config = AutoConfig.from_pretrained(model_id)
            model_cls = MODEL_TYPE_TO_CLS_MAPPING.get(getattr(config, "model_type", None))
            if model_cls is not None:
                self._model = model_cls.from_pretrained(model_id, device=device)
            else:
                self._model = OVModelForCausalLM.from_pretrained(
                    model_id, from_onnx=False, device=device
                )
            print("[minutes] openvino model loaded", flush=True)
        else:
            print("[minutes] loading torch model...", flush=True)
            self._model = AutoModelForCausalLM.from_pretrained(model_id)
            self._model = self._model.to("cpu")
            self._model.eval()
            print("[minutes] torch model loaded", flush=True)

    def summarize(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""
        print(f"[minutes] summarize start chars={len(cleaned)}", flush=True)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that writes concise meeting minutes in Japanese. "
                    "Follow the exact template provided and do not add or remove sections. "
                    "Do not include the original transcript or direct quotes."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize the following meeting transcript into concise minutes. "
                    "Use the template below exactly, filling in each section. "
                    "Keep it short and do not include the original text or direct quotes.\n\n"
                    "Template:\n"
                    "■ 会議名：\n"
                    "■ 日時：\n"
                    "■ 出席者：\n"
                    "■ 目的：\n"
                    "\n"
                    "■ 議題\n"
                    "・\n"
                    "・\n"
                    "\n"
                    "■ 決定事項\n"
                    "・\n"
                    "・\n"
                    "\n"
                    "■ ToDo\n"
                    "・内容 / 担当 / 期限\n"
                    "・\n"
                    "\n"
                    "■ 論点・メモ\n"
                    "・\n"
                    "\n"
                    "■ 未決事項\n"
                    "・\n"
                    "\n"
                    "■ 次回予定\n"
                    "・\n\n"
                    f"{cleaned}"
                ),
            },
        ]
        if hasattr(self._tokenizer, "apply_chat_template"):
            print("[minutes] using chat template", flush=True)
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            print("[minutes] using fallback prompt", flush=True)
            prompt = (
                "You are a helpful assistant that writes concise meeting minutes in Japanese. "
                "Follow the exact template provided and do not add or remove sections. "
                "Do not include the original transcript or direct quotes.\n\n"
                "Summarize the following meeting transcript into concise minutes. "
                "Use the template below exactly, filling in each section. "
                "Keep it short and do not include the original text or direct quotes.\n\n"
                "Template:\n"
                "■ 会議名：\n"
                "■ 日時：\n"
                "■ 出席者：\n"
                "■ 目的：\n"
                "\n"
                "■ 議題\n"
                "・\n"
                "・\n"
                "\n"
                "■ 決定事項\n"
                "・\n"
                "・\n"
                "\n"
                "■ ToDo\n"
                "・内容 / 担当 / 期限\n"
                "・\n"
                "\n"
                "■ 論点・メモ\n"
                "・\n"
                "\n"
                "■ 未決事項\n"
                "・\n"
                "\n"
                "■ 次回予定\n"
                "・\n\n"
                f"{cleaned}\n\nMinutes:"
            )
        print(f"[minutes] prompt chars={len(prompt)}", flush=True)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        try:
            input_len = inputs["input_ids"].shape[-1]
        except Exception:
            input_len = None
        print(f"[minutes] tokenized input_len={input_len}", flush=True)
        print("[minutes] generate start", flush=True)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=MINUTES_MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.2,
            )
        print("[minutes] generate done", flush=True)
        decoded = self._tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"[minutes] decoded chars={len(decoded)}", flush=True)
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt) :].strip()
        else:
            decoded = decoded.strip()

        # Strip any echoed prompt/system text and keep only the minutes section.
        decoded = _strip_prompt_echo(decoded)
        marker = "会議議事録"
        if marker in decoded:
            decoded = decoded[decoded.index(marker) :]
        if not decoded.startswith("**会議議事録**"):
            decoded = f"**会議議事録**\n\n{decoded}"
        return decoded.strip()


def _strip_prompt_echo(text: str) -> str:
    if not text:
        return ""
    lines = [line.rstrip() for line in text.splitlines()]
    if not lines:
        return text.strip()
    # Drop leading chat labels like "user"/"system" and remove everything up to "model".
    lowered = [line.strip().lower() for line in lines]
    if "model" in lowered:
        model_idx = lowered.index("model")
        return "\n".join(lines[model_idx + 1 :]).strip()
    for idx, line in enumerate(lowered):
        if line in ("user", "system", "assistant"):
            continue
        if idx > 0:
            return "\n".join(lines[idx:]).strip()
    return "\n".join(lines).strip()


def get_minutes_summarizer() -> MinutesSummarizer:
    global _SUMMARIZER
    if _SUMMARIZER is None:
        _SUMMARIZER = MinutesSummarizer()
    return _SUMMARIZER


def _read_text_from_path(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _print_usage() -> None:
    print(
        "Usage: python minutes/summarizer.py [--file PATH] [--max-chars N]",
        flush=True,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run minutes summarization from a text file or stdin."
    )
    parser.add_argument(
        "--file",
        dest="file_path",
        help="Path to input text file. If omitted, reads stdin.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=MINUTES_MAX_INPUT_CHARS,
        help="Max input characters before truncation.",
    )
    args = parser.parse_args()

    if args.max_chars <= 0:
        print("max-chars must be positive.", flush=True)
        return 2

    if args.file_path:
        print(f"[minutes] cli file={args.file_path}", flush=True)
        text = _read_text_from_path(args.file_path)
    else:
        try:
            text = input()
        except EOFError:
            text = ""

    text = (text or "").strip()
    if not text:
        print("No input text.", flush=True)
        return 1

    if len(text) > args.max_chars:
        text = text[: args.max_chars]

    summarizer = get_minutes_summarizer()
    result = summarizer.summarize(text)
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
