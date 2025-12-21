"""
Bootstrap script for NBA Analytics v3.

Recreates the required folder structure ONLY if folders do not exist.
Never overwrites or recreates existing directories.

Usage:
    python bootstrap_structure.py
"""

from pathlib import Path

# IMPORTANT FIX:
# Move one directory up so PROJECT_ROOT = project root, not src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -------------------------------------------------------------------
# Directory structure for the entire project
# -------------------------------------------------------------------
DIRS = [
    # Source code
    "src",
    "src/ingestion",
    "src/features",
    "src/model",
    "src/ranking",
    "src/monitoring",
    "src/pipeline",
    # Data directories
    "data",
    "data/raw",
    "data/ingestion",
    "data/parquet",
    "data/predictions",
    "data/rankings",
    "data/drift",
    # Model registry
    "models",
    "models/registry",
]

# Optional placeholder files to keep empty dirs in Git
PLACEHOLDERS = [
    "data/raw/.gitkeep",
    "data/ingestion/.gitkeep",
    "data/parquet/.gitkeep",
    "data/predictions/.gitkeep",
    "data/rankings/.gitkeep",
    "data/drift/.gitkeep",
    "models/registry/.gitkeep",
]


def main():
    print("Bootstrapping NBA Analytics v3 folder structure...")

    # Create directories only if missing
    for d in DIRS:
        path = PROJECT_ROOT / d
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"CREATED DIR   {d}")
        else:
            print(f"EXISTS        {d}")

    # Create placeholder files only if missing
    for f in PLACEHOLDERS:
        path = PROJECT_ROOT / f
        if not path.exists():
            path.touch()
            print(f"CREATED FILE  {f}")
        else:
            print(f"EXISTS        {f}")

    print("Bootstrap complete.")


if __name__ == "__main__":
    main()
