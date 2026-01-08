import os
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

try:
    from pyannote.audio import Pipeline
except Exception:  # pragma: no cover - import guard for optional dependency
    Pipeline = None


@dataclass(frozen=True)
class DiarizationConfig:
    model_id: str
    auth_token: Optional[str]
    device: str


class PyannoteDiarizer:
    def __init__(
        self,
        model_id: str = "pyannote/speaker-diarization-community-1",
        auth_token: Optional[str] = None,
        device: str = "cpu",
    ):
        if Pipeline is None:
            raise RuntimeError("pyannote.audio is not installed.")
        token = auth_token or os.getenv("PYANNOTE_AUTH_TOKEN")
        if not token:
            raise RuntimeError("Set PYANNOTE_AUTH_TOKEN to use speaker diarization.")

        # Support different pyannote.audio versions (token arg name changed).
        try:
            self._pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)
        except TypeError:
            self._pipeline = Pipeline.from_pretrained(model_id, token=token)
        self._lock = threading.Lock()
        try:
            self._pipeline.to(device)
        except Exception:
            # Some pipeline versions do not expose `.to`; ignore device hint.
            pass

    def dominant_speaker(
        self,
        audio: np.ndarray,
        sr: int,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> Optional[str]:
        if audio is None or sr <= 0 or audio.size == 0:
            return None
        # Skip very short chunks to avoid unstable stats in the model.
        if audio.size < int(sr * 0.5):
            return None

        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        with self._lock:
            diarization = self._pipeline(
                {"waveform": waveform, "sample_rate": sr},
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        annotation = diarization
        if hasattr(diarization, "speaker_diarization"):
            annotation = diarization.speaker_diarization
        elif hasattr(diarization, "annotation"):
            annotation = diarization.annotation
        elif isinstance(diarization, dict):
            if "speaker_diarization" in diarization:
                annotation = diarization["speaker_diarization"]
            elif "annotation" in diarization:
                annotation = diarization["annotation"]
        durations: dict[str, float] = {}
        last_end: dict[str, float] = {}
        if hasattr(annotation, "itertracks"):
            iterator = annotation.itertracks(yield_label=True)
        elif hasattr(annotation, "itersegments"):
            # Fallback for older core objects: no track info, only segments + label.
            iterator = ((segment, None, label) for segment, label in annotation.itersegments(yield_label=True))
        else:
            return None
        for segment, _, label in iterator:
            durations[label] = durations.get(label, 0.0) + segment.duration
            last_end[label] = max(last_end.get(label, 0.0), segment.end)

        if not durations:
            return None
        # Prefer the longest speaker; break ties by most recent activity.
        return max(durations.keys(), key=lambda k: (durations[k], last_end.get(k, 0.0)))
