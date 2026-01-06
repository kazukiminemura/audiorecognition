from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class SpeechRecognizer(Protocol):
    def transcribe_array(self, audio, sr) -> str:
        ...


@runtime_checkable
class Translator(Protocol):
    def translate(self, text: str) -> str:
        ...


@dataclass(slots=True)
class SpeechToEnglishPipeline:
    recognizer: SpeechRecognizer
    translator: Translator

    def run(self, audio, sr) -> str:
        jp_text = self.recognizer.transcribe_array(audio, sr)
        return self.translator.translate(jp_text) if jp_text else ""
