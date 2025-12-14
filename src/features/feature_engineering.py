# ============================================================
# File: src/features/feature_engineering.py
# Purpose: Generate features parquet + sanity stats
# ============================================================

import logging
import pandas as pd
from pathlib import Path


def generate_features():
    """
    Core feature engineering logic.
    Produces features + target column WIN (1 = win, 0 = loss).
    """
    logging.info("Starting feature engineering...")

    # Example: load raw schedule data
    raw_file = Path("data/raw/schedule.csv")
    if not raw_file.exists():
        logging.error("Missing raw schedule file at %s", raw_file)
        return

    df = pd.read_csv(raw_file)

    # --- Create target column ---
    # Assumes raw data has a "WL" column with values "W" or "L"
    if "WL" in df.columns:
        df["WIN"] = df["WL"].eq("W").astype(int)
    else:
        logging.error(
            "No WL column found in raw data. Cannot create WIN target column."
        )
        return

    # --- Example feature engineering ---
    df["is_home"] = df["MATCHUP"].str.contains("vs")

    # Save parquet
    out_file = Path("data/cache/features_full.parquet")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file, index=False)

    # Save sanity stats
    stats_file = Path("data/results/feature_stats.csv")
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    df.describe().to_csv(stats_file)

    logging.info(
        "Feature engineering complete. Outputs saved to %s and %s",
        out_file,
        stats_file,
    )


def main():
    """Entry point for console script `nba-features`."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    generate_features()


if __name__ == "__main__":
    main()
