from src import hf_env  # ensure HF env defaults before other imports
import os

from huggingface_hub import snapshot_download
from silero_vad import load_silero_vad

from src.settings import (
    DEFAULT_MODEL_ID,
    MINUTES_MODEL_ID,
    PROJECT_MODELS_DIR,
    TRANSLATION_MODEL_ID,
    LFM2_AUDIO_REPO,
)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) not in ("0", "false", "False", "no", "NO")


def _preload_vad(progress_cb=None, is_cancelled=None) -> bool:
    if is_cancelled and is_cancelled():
        return False
    if progress_cb:
        progress_cb("Downloading VAD model...")
    os.environ.setdefault("TORCH_HOME", PROJECT_MODELS_DIR)
    model = load_silero_vad()
    model = model.to("cpu")
    model.eval()
    return True


def preload_models(
    progress_cb=None,
    is_cancelled=None,
    *,
    include_minutes=None,
    include_lfm2=None,
    include_vad=None,
):
    os.environ.setdefault("HF_HOME", PROJECT_MODELS_DIR)
    # Re-assert in case callers changed env after import.
    hf_env._init_hf_env()

    def step(label: str, model_id: str):
        if is_cancelled and is_cancelled():
            return False
        if progress_cb:
            progress_cb(label)
        snapshot_download(
            repo_id=model_id,
            cache_dir=PROJECT_MODELS_DIR,
            local_dir_use_symlinks=False,
        )
        return True

    if include_minutes is None:
        include_minutes = _env_flag("PRELOAD_MINUTES_MODEL", "0")
    if include_lfm2 is None:
        include_lfm2 = _env_flag("PRELOAD_LFM2_MODEL", "0")
    if include_vad is None:
        include_vad = _env_flag("PRELOAD_VAD_MODEL", "0")

    models = [
        ("Downloading Whisper model...", DEFAULT_MODEL_ID),
        ("Downloading translation model...", TRANSLATION_MODEL_ID),
    ]
    if include_lfm2:
        models.append(("Downloading LFM2 audio model...", LFM2_AUDIO_REPO))
    if include_minutes:
        models.append(("Downloading minutes summarizer...", MINUTES_MODEL_ID))

    for label, model_id in models:
        if not step(label, model_id):
            return False

    if include_vad and not _preload_vad(progress_cb=progress_cb, is_cancelled=is_cancelled):
        return False

    return True
