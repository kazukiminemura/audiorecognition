import os

from huggingface_hub import snapshot_download

from settings import (
    DEFAULT_MODEL_ID,
    MINUTES_MODEL_ID,
    PROJECT_MODELS_DIR,
    TRANSLATION_MODEL_ID,
    LFM2_AUDIO_REPO,
)


def preload_models(progress_cb=None, is_cancelled=None):
    os.environ.setdefault("HF_HOME", PROJECT_MODELS_DIR)

    def step(label: str, model_id: str):
        if is_cancelled and is_cancelled():
            return False
        if progress_cb:
            progress_cb(label)
        snapshot_download(repo_id=model_id, cache_dir=PROJECT_MODELS_DIR)
        return True

    models = [
        ("Downloading Whisper model...", DEFAULT_MODEL_ID),
        ("Downloading translation model...", TRANSLATION_MODEL_ID),
        ("Downloading LFM2 audio model...", LFM2_AUDIO_REPO),
        ("Downloading minutes summarizer...", MINUTES_MODEL_ID),
    ]

    for label, model_id in models:
        if not step(label, model_id):
            return False

    return True
