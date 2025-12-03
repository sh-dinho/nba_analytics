# ============================================================
# File: scripts/download_dataset.py
# Purpose: Download NBA games dataset and save to data folder
# ============================================================

import os
import argparse
import pandas as pd
from datasets import load_dataset

# Paths
BASE_DATA_DIR = os.path.join("C:/Users/Mohamadou/nba_analytics", "data")

def save_split(dataset, split_name, config_name):
    """Save a dataset split to CSV with diagnostics."""
    df = dataset[split_name].to_pandas()
    out_file = os.path.join(BASE_DATA_DIR, f"{config_name}_{split_name}.csv")

    if os.path.exists(out_file):
        print(f"⚠️ Overwriting existing file: {out_file}")

    df.to_csv(out_file, index=False)
    print(f"✅ {config_name}/{split_name} saved to {out_file} ({len(df)} rows)")

    if "homewin" in df.columns:
        print(f"📊 {config_name}/{split_name} label distribution:", df["homewin"].value_counts().to_dict())

def get_revision_safe(dataset, split="train"):
    """Safely get dataset revision or version."""
    info = dataset[split].info
    revision = getattr(info, "dataset_revision", None)
    if revision is None:
        revision = getattr(info, "version", "unknown")
    return str(revision)

def main(force_refresh=False, configs=None):
    os.makedirs(BASE_DATA_DIR, exist_ok=True)

    # Default configs if none specified
    if configs is None:
        configs = ["games", "games_details", "players", "ranking", "teams"]

    for config in configs:
        print(f"📥 Checking NBA dataset config: {config}...")
        try:
            dataset = load_dataset("hamzas/nba-games", config)

            # Expected local files
            expected_files = {f"{config}_{split}.csv" for split in dataset.keys()}
            local_files = set(os.listdir(BASE_DATA_DIR))

            # Safe version check
            revision = get_revision_safe(dataset)
            version_file = os.path.join(BASE_DATA_DIR, f"{config}_version.txt")

            local_version = None
            if os.path.exists(version_file):
                with open(version_file, "r") as f:
                    local_version = f.read().strip()

            # Skip if already up-to-date
            if expected_files.issubset(local_files) and local_version == revision and not force_refresh:
                print(f"✅ {config} dataset is already up-to-date (version {local_version}). Skipping re-download.")
                continue

            # Otherwise refresh
            print(f"🔄 Downloading/refreshing {config} dataset (version {revision})...")
            for split in dataset.keys():
                save_split(dataset, split, config)

            # Save version info
            with open(version_file, "w") as f:
                f.write(str(revision))

        except Exception as e:
            print(f"❌ Error with config {config}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NBA dataset from Hugging Face")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh even if dataset already exists locally"
    )
    parser.add_argument(
        "--config",
        nargs="+",
        choices=["games", "games_details", "players", "ranking", "teams"],
        help="Specify one or more configs to download (default: all)"
    )
    args = parser.parse_args()
    main(force_refresh=args.refresh, configs=args.config)