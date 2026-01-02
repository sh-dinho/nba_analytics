from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Validate Features
# File: src/scripts/validate_features.py
# Author: Sadiq
#
# Description:
#     Validates the canonical feature snapshot against FeatureRow.
#     Checks:
#       • Schema drift (missing/extra columns)
#       • Row-level validation via Pydantic
#       • NaN distribution
#       • Summary of validation failures
# ============================================================

import pandas as pd
from loguru import logger

from src.config.paths import FEATURES_SNAPSHOT
from src.features.feature_schema import FeatureRow


def validate_features(max_errors: int = 20):
    """
    Audit the feature snapshot for model-readiness.
    """
    logger.info("=== 🛡️ Validating FEATURES_SNAPSHOT ===")

    if not FEATURES_SNAPSHOT.exists():
        logger.error(f"❌ Missing file: {FEATURES_SNAPSHOT}")
        return

    df = pd.read_parquet(FEATURES_SNAPSHOT)
    total_rows = len(df)

    logger.info(f"📊 Loaded {total_rows} rows across {len(df.columns)} columns.")

    # --------------------------------------------------------
    # 1. Schema Drift Detection
    # --------------------------------------------------------
    expected_fields = set(FeatureRow.model_fields.keys())
    actual_fields = set(df.columns)

    missing = expected_fields - actual_fields
    extra = actual_fields - expected_fields

    if missing:
        logger.warning(f"⚠️ SCHEMA DRIFT — Missing expected columns: {missing}")

    if extra:
        logger.info(f"ℹ️ Extra columns found (ignored by model): {extra}")

    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # --------------------------------------------------------
    # 2. Null Value Distribution
    # --------------------------------------------------------
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]

    if not nan_cols.empty:
        logger.warning("⚠️ NaN values detected:")
        for col, count in nan_cols.items():
            pct = (count / total_rows) * 100
            logger.warning(f"  • {col}: {count} rows ({pct:.2f}%)")
    else:
        logger.info("✨ No NaN values detected.")

    # --------------------------------------------------------
    # 3. Row-Level Deep Validation (Pydantic)
    # --------------------------------------------------------
    logger.info("🔍 Running row-level schema validation...")

    error_count = 0
    sample_errors = []

    for idx, row in df.iterrows():
        try:
            FeatureRow(**row.to_dict())
        except Exception as e:
            error_count += 1
            if len(sample_errors) < max_errors:
                sample_errors.append(f"Row {idx}: {e}")

    # --------------------------------------------------------
    # 4. Summary
    # --------------------------------------------------------
    if error_count > 0:
        logger.error(f"❌ Validation failed for {error_count} rows.")
        logger.error("Sample errors:")
        for err in sample_errors:
            logger.error(f"  - {err}")
    else:
        logger.success("✅ All rows passed Pydantic schema validation.")



if __name__ == "__main__":
    validate_features()
