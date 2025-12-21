"""
Cleanup script for NBA Analytics v3.

This script DELETES everything inside src/ EXCEPT the official project modules.
It keeps only the files that are part of the NBA Analytics v3 application.

Usage:
    python clean_src.py

WARNING:
    This is destructive. Make sure your repo is backed up or committed.
"""

from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

# -------------------------------------------------------------------
# WHITELIST: Only these paths inside src/ will be kept.
# Everything else in src/ will be deleted.
# -------------------------------------------------------------------
KEEP_PATHS = {
    # ingestion
    "src/ingestion",
    "src/ingestion/ingestion.py",
    "src/ingestion/ingestion_runner.py",
    "src/ingestion/load_raw_schedule.py",
    "src/ingestion/normalize_schema.py",
    # features
    "src/features",
    "src/features/feature_builder.py",
    "src/features/feature_store.py",
    "src/features/feature_store_runner.py",
    # model
    "src/model",
    "src/model/registry.py",
    "src/model/train.py",
    "src/model/train_runner.py",
    "src/model/predict.py",
    "src/model/predict_runner.py",
    # ranking
    "src/ranking",
    "src/ranking/ranking_runner.py",
    # monitoring
    "src/monitoring",
    "src/monitoring/drift_detector.py",
    # pipeline
    "src/pipeline",
    "src/pipeline/pipeline_runner.py",
}

KEEP_ABS = {(PROJECT_ROOT / p).resolve() for p in KEEP_PATHS}


def should_keep(path: Path) -> bool:
    """Return True if path should be kept."""
    path = path.resolve()

    # Exact match
    if path in KEEP_ABS:
        return True

    # Keep parent directories of whitelisted files
    for keep in KEEP_ABS:
        try:
            keep.relative_to(path)
            return True
        except ValueError:
            continue

    return False


def main():
    print(f"Cleaning src/ at: {SRC_ROOT}")
    print("WARNING: This will DELETE everything in src/ not explicitly whitelisted.")
    confirm = input("Type DELETE to continue: ").strip()

    if confirm != "DELETE":
        print("Aborted.")
        return

    for item in SRC_ROOT.iterdir():
        if should_keep(item):
            print(f"KEEP   {item.relative_to(PROJECT_ROOT)}")
            continue

        # Delete file or directory
        if item.is_dir():
            print(f"DELETE DIR  {item.relative_to(PROJECT_ROOT)}")
            shutil.rmtree(item)
        else:
            print(f"DELETE FILE {item.relative_to(PROJECT_ROOT)}")
            item.unlink()

    print("src/ cleanup complete.")


if __name__ == "__main__":
    main()
