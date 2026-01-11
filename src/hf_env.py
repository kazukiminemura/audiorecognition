import os


def _init_hf_env() -> None:
    # Default to safe HF settings on Windows (no symlinks / no hf_xet).
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _disable_hf_symlinks() -> None:
    # Force-disable symlinks in huggingface_hub regardless of OS support.
    try:
        from huggingface_hub import file_download as fd
    except Exception:
        return

    def _always_false(cache_dir=None) -> bool:  # pragma: no cover - trivial
        return False

    fd.are_symlinks_supported = _always_false
    fd._are_symlinks_supported_in_dir.clear()


_init_hf_env()
_disable_hf_symlinks()
