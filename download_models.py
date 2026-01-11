import hf_env  # ensure HF env defaults before other imports
import argparse

from model_preload import preload_models


def main():
    parser = argparse.ArgumentParser(
        description="Download models into the local cache without starting the UI."
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Download only core models (Whisper + translation).",
    )
    args = parser.parse_args()

    include_extras = not args.core_only
    ok = preload_models(
        include_minutes=include_extras, include_lfm2=include_extras
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
