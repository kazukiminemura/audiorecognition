import html
from dataclasses import dataclass
from typing import Optional


@dataclass
class SpeakerSegment:
    text: str
    speaker: Optional[str]


class SpeakerTextAccumulator:
    def __init__(self):
        self._segments: list[SpeakerSegment] = []

    def append(self, text: str, speaker: Optional[str]):
        self._segments.append(SpeakerSegment(text=text, speaker=speaker))

    def replace_last(self, text: str, speaker: Optional[str]):
        if self._segments:
            self._segments[-1] = SpeakerSegment(text=text, speaker=speaker)
        else:
            self._segments.append(SpeakerSegment(text=text, speaker=speaker))

    def render_plain(self) -> str:
        return "\n".join(seg.text for seg in self._segments)

    def render_html(self, palette: "SpeakerPalette") -> str:
        lines: list[str] = []
        for seg in self._segments:
            if not seg.text:
                continue
            speaker = seg.speaker or "Unknown"
            color = palette.color_for(speaker)
            label = palette.label_for(speaker)
            safe_text = html.escape(seg.text)
            safe_label = html.escape(label)
            lines.append(
                f'<span style="color:{color}; font-weight:600;">[{safe_label}]</span> '
                f'<span style="color:{color};">{safe_text}</span>'
            )
        return "<br/>".join(lines)


class SpeakerPalette:
    def __init__(self):
        self._colors = [
            "#005a9e",
            "#9f4d00",
            "#2d6a4f",
            "#6a4c93",
            "#b00020",
            "#9c4f19",
            "#0f4c5c",
        ]
        self._labels: dict[str, str] = {}
        self._colors_by_speaker: dict[str, str] = {}
        self._counter = 0

    def label_for(self, speaker: str) -> str:
        if speaker == "Unknown":
            return "UNK"
        if speaker not in self._labels:
            self._labels[speaker] = f"S{len(self._labels) + 1}"
        return self._labels[speaker]

    def color_for(self, speaker: str) -> str:
        if speaker == "Unknown":
            return "#6b7280"
        if speaker not in self._colors_by_speaker:
            color = self._colors[self._counter % len(self._colors)]
            self._colors_by_speaker[speaker] = color
            self._counter += 1
        return self._colors_by_speaker[speaker]
