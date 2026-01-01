from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Verify Canonical Features
# File: src/scripts/verify_features.py
# Author: Sadiq
#
# Description:
#     Validates canonical features:
#       • Column integrity vs FeatureBuilder schema
#       • Missing/extra columns
#       • NaN distribution
#       • Numeric sanity checks
#       • Row-level validation (optional)
#       • Latest 5 entries preview
#
#     Works with:
#       • Feature snapshot (if exists)
#       • OR dynamic feature build from LONG_SNAPSHOT
# ============================================================

import pandas as pd
from loguru import logger

from src.config.paths import FEATURES_SNAPSHOT, LONG_SNAPSHOT
from src.features.builder import FeatureBuilder


def _load_features(feature_version: str) -> pd.DataFrame:
    """Load feature snapshot if available, otherwise build dynamically."""
    if FEATURES_SNAPSHOT.exists():
        logger.info(f"Loading feature snapshot: {FEATURES_SNAPSHOT}")
        return pd.read_parquet(FEATURES_SNAPSHOT)

    if not LONG_SNAPSHOT.exists():
        raise RuntimeError("No feature snapshot and no LONG_SNAPSHOT available.")

    logger.info("Building features dynamically from LONG_SNAPSHOT...")
    long_df = pd.read_parquet(LONG_SNAPSHOT)
    fb = FeatureBuilder(version=feature_version)
    return fb.build(long_df)


def verify_features(feature_version: str = "v1", sample_size: int = 5000):
    print("\n=== VERIFYING CANONICAL FEATURES ===")

    try:
        df = _load_features(feature_version)
    except Exception as e:
        logger.error(f"Failed to load or build features: {e}")
        return

    print(f"Total Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # 1. Schema Integrity
    # --------------------------------------------------------
    fb = FeatureBuilder(version=feature_version)
    expected_cols = set(fb.expected_feature_columns())
    actual_cols = set(df.columns)

    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols

    if missing:
        logger.error(f"❌ Missing columns: {missing}")
    else:
        logger.success("✅ All expected columns present.")

    if extra:
        logger.warning(f"ℹ️ Extra columns found: {extra}")

    # --------------------------------------------------------
    # 2. NaN Distribution
    # --------------------------------------------------------
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]

    if not nan_cols.empty:
        print("\nColumns with NaNs:")
        print(nan_cols.sort_values(ascending=False).head(20))
    else:
        logger.success("✅ No NaNs detected.")

    # --------------------------------------------------------
    # 3. Numeric sanity checks
    # --------------------------------------------------------
    numeric_cols = df.select_dtypes(include=["number"]).columns

    print("\n--- Numeric Sanity Checks ---")
    for col in numeric_cols:
        if df[col].abs().max() > 1e9:
            logger.warning(f"⚠️ Suspicious magnitude in {col}: max={df[col].abs().max()}")

    # --------------------------------------------------------
    # 4. Sample row validation (optional)
    # --------------------------------------------------------
    sample = df.sample(min(sample_size, len(df)))
    print(f"\nValidating sample of {len(sample)} rows...")

    # If you later add a Pydantic schema, validate here.
    # For now, just ensure no catastrophic issues.
    if sample.isna().all(axis=1).any():
        logger.error("❌ Some sampled rows are entirely NaN.")
    else:
        logger.success("✅ Sample rows appear structurally valid.")

    # --------------------------------------------------------
    # 5. Preview latest rows
    # --------------------------------------------------------
    print("\n--- Latest 5 Rows ---")
    print(df.sort_values("date").tail(5))

    print("\n=== DONE ===")


if __name__ == "__main__":
    verify_features()