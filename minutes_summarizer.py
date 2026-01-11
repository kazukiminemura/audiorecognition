import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from settings import MINUTES_MAX_NEW_TOKENS, MINUTES_MODEL_ID, PROJECT_MODELS_DIR

_SUMMARIZER = None


class MinutesSummarizer:
    def __init__(self, model_id: str = MINUTES_MODEL_ID):
        os.environ.setdefault("HF_HOME", PROJECT_MODELS_DIR)
        self._model_id = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id)
        self._model = self._model.to("cpu")
        self._model.eval()

    def summarize(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that writes concise meeting minutes in Japanese.",
            },
            {
                "role": "user",
                "content": (
                    "Summarize the following meeting transcript into concise minutes. "
                    "Use bullet points and keep it short.\n\n"
                    f"{cleaned}"
                ),
            },
        ]
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = (
                "You are a helpful assistant that writes concise meeting minutes in Japanese.\n\n"
                "Summarize the following meeting transcript into concise minutes. "
                "Use bullet points and keep it short.\n\n"
                f"{cleaned}\n\nMinutes:"
            )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=MINUTES_MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.2,
            )
        decoded = self._tokenizer.decode(output[0], skip_special_tokens=True)
        if decoded.startswith(prompt):
            return decoded[len(prompt) :].strip()
        return decoded.strip()


def get_minutes_summarizer() -> MinutesSummarizer:
    global _SUMMARIZER
    if _SUMMARIZER is None:
        _SUMMARIZER = MinutesSummarizer()
    return _SUMMARIZER
