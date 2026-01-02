from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Verify Canonical Features
# File: src/scripts/verify_features.py
# Author: Sadiq
# ============================================================

import pandas as pd
from loguru import logger

from src.config.paths import FEATURES_SNAPSHOT, LONG_SNAPSHOT
from src.features.builder import FeatureBuilder


# ------------------------------------------------------------
# Load Features (Snapshot or Dynamic)
# ------------------------------------------------------------
def _load_features() -> pd.DataFrame:
    """Load feature snapshot if available, otherwise build dynamically."""
    # 1. Snapshot exists
    if FEATURES_SNAPSHOT.exists():
        logger.info(f"Loading feature snapshot: {FEATURES_SNAPSHOT}")

        if FEATURES_SNAPSHOT.is_file():
            return pd.read_parquet(FEATURES_SNAPSHOT)

        # Directory of parquet parts
        parts = list(FEATURES_SNAPSHOT.glob("**/*.parquet"))
        if not parts:
            raise RuntimeError(f"FEATURES_SNAPSHOT directory is empty: {FEATURES_SNAPSHOT}")

        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)

    # 2. Fallback → dynamic build
    if not LONG_SNAPSHOT.exists():
        raise RuntimeError("No feature snapshot and no LONG_SNAPSHOT available.")

    logger.info("Building features dynamically from LONG_SNAPSHOT...")
    long_df = pd.read_parquet(LONG_SNAPSHOT)

    fb = FeatureBuilder()  # version‑agnostic
    return fb.build(long_df)


# ------------------------------------------------------------
# Main Verification Routine
# ------------------------------------------------------------
def verify_features(sample_size: int = 5000):
    print("\n=== VERIFYING CANONICAL FEATURES ===")

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------
    try:
        df = _load_features()
    except Exception as e:
        logger.error(f"Failed to load or build features: {e}")
        return

    print(f"Total Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # 1. Schema Integrity
    # --------------------------------------------------------
    fb = FeatureBuilder()
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
    # 3. Numeric Sanity Checks
    # --------------------------------------------------------
    numeric_cols = df.select_dtypes(include=["number"]).columns

    print("\n--- Numeric Sanity Checks ---")
    for col in numeric_cols:
        col_max = df[col].abs().max()

        if pd.isna(col_max):
            logger.warning(f"⚠️ Column {col} contains only NaN values.")
            continue

        if col_max > 1e9:
            logger.warning(f"⚠️ Suspicious magnitude in {col}: max={col_max}")

        if df[col].nunique() == 1:
            logger.info(f"ℹ️ Column {col} is constant (nunique=1).")

        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"⚠️ Column {col} is not numeric dtype.")

    # --------------------------------------------------------
    # 4. Sample Row Validation
    # --------------------------------------------------------
    sample = df.sample(min(sample_size, len(df)))
    print(f"\nValidating sample of {len(sample)} rows...")

    if sample.isna().all(axis=1).any():
        logger.error("❌ Some sampled rows are entirely NaN.")
    else:
        logger.success("✅ Sample rows appear structurally valid.")

    # --------------------------------------------------------
    # 5. Preview Latest Rows
    # --------------------------------------------------------
    print("\n--- Latest 5 Rows ---")
    if "date" in df.columns:
        print(df.sort_values("date").tail(5))
    else:
        logger.warning("⚠️ No 'date' column found — cannot preview latest rows.")
        print(df.tail(5))

    print("\n=== DONE ===")


if __name__ == "__main__":
    verify_features()
