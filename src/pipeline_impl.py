import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.pipeline import SpeechToEnglishPipeline
from src.settings import (
    PROJECT_MODELS_DIR,
    TRANSLATION_MODEL_ID,
    TRANSLATION_MODEL_ID_EN_JA,
)

_TRANSLATION_CACHE: dict[str, tuple[AutoTokenizer, AutoModelForSeq2SeqLM]] = {}


class JaToEnTranslator:
    def __init__(self, model_id: str = TRANSLATION_MODEL_ID):
        try:
            import sentencepiece  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "sentencepiece is required for the translation tokenizer. "
                "Install it with: pip install sentencepiece"
            ) from exc
        xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
        self.device = torch.device("xpu" if xpu_available else "cpu")
        self.tokenizer, self.model = _load_translation_model(
            [model_id, "Helsinki-NLP/opus-mt-ja-en"],
            self.device,
        )
        self.src_lang = "jpn_Jpan"
        self.tgt_lang = "eng_Latn"

    def translate(self, text: str) -> str:
        if not text:
            return ""
        _set_nllb_langs(self.tokenizer, self.src_lang, self.tgt_lang)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        forced_bos = _get_forced_bos_id(self.tokenizer, self.tgt_lang)
        output = self.model.generate(
            **inputs,
            max_new_tokens=256,
            forced_bos_token_id=forced_bos,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


class EnToJaTranslator:
    def __init__(self, model_id: str = TRANSLATION_MODEL_ID_EN_JA):
        try:
            import sentencepiece  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "sentencepiece is required for the translation tokenizer. "
                "Install it with: pip install sentencepiece"
            ) from exc
        xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
        self.device = torch.device("xpu" if xpu_available else "cpu")
        self.tokenizer, self.model = _load_translation_model(
            [model_id, "Helsinki-NLP/opus-mt-en-ja"],
            self.device,
        )
        self.src_lang = "eng_Latn"
        self.tgt_lang = "jpn_Jpan"

    def translate(self, text: str) -> str:
        if not text:
            return ""
        _set_nllb_langs(self.tokenizer, self.src_lang, self.tgt_lang)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        forced_bos = _get_forced_bos_id(self.tokenizer, self.tgt_lang)
        output = self.model.generate(
            **inputs,
            max_new_tokens=256,
            forced_bos_token_id=forced_bos,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def build_speech_to_english_pipeline(recognizer) -> SpeechToEnglishPipeline:
    return SpeechToEnglishPipeline(recognizer=recognizer, translator=JaToEnTranslator())


def _load_translation_model(model_ids: list[str], device: torch.device):
    last_exc: Exception | None = None
    os.environ.setdefault("HF_HOME", PROJECT_MODELS_DIR)
    for candidate in model_ids:
        if not candidate:
            continue
        try:
            cached = _TRANSLATION_CACHE.get(candidate)
            if cached is not None:
                tokenizer, model = cached
            else:
                tokenizer = AutoTokenizer.from_pretrained(candidate)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    candidate,
                    attn_implementation="eager",
                ).to(device)
                _TRANSLATION_CACHE[candidate] = (tokenizer, model)
            return tokenizer, model
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(
        "Failed to load translation model. Tried: "
        + ", ".join([m for m in model_ids if m])
    ) from last_exc


def _set_nllb_langs(tokenizer, src_lang: str, tgt_lang: str) -> None:
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_lang
    if hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = tgt_lang


def _get_forced_bos_id(tokenizer, tgt_lang: str):
    if hasattr(tokenizer, "lang_code_to_id"):
        value = tokenizer.lang_code_to_id.get(tgt_lang)
        if value is not None:
            return value
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        value = tokenizer.convert_tokens_to_ids(tgt_lang)
        if isinstance(value, int) and value >= 0:
            return value
    return None
