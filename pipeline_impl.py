from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from pipeline import SpeechToEnglishPipeline
from settings import TRANSLATION_MODEL_ID, TRANSLATION_MODEL_ID_EN_JA


class JaToEnTranslator:
    def __init__(self, model_id: str = TRANSLATION_MODEL_ID):
        try:
            import sentencepiece  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "sentencepiece is required for the translation tokenizer. "
                "Install it with: pip install sentencepiece"
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    def translate(self, text: str) -> str:
        if not text:
            return ""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        output = self.model.generate(**inputs, max_new_tokens=256)
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    def translate(self, text: str) -> str:
        if not text:
            return ""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        output = self.model.generate(**inputs, max_new_tokens=256)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def build_speech_to_english_pipeline(recognizer) -> SpeechToEnglishPipeline:
    return SpeechToEnglishPipeline(recognizer=recognizer, translator=JaToEnTranslator())
