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
    logger.info("=== Validating FEATURES_SNAPSHOT ===")

    if not FEATURES_SNAPSHOT.exists():
        logger.error("FEATURES_SNAPSHOT does not exist.")
        return

    df = pd.read_parquet(FEATURES_SNAPSHOT)
    logger.info(f"Loaded {len(df)} feature rows.")

    # --------------------------------------------------------
    # 1. Schema drift detection
    # --------------------------------------------------------
    expected_fields = set(FeatureRow.model_fields.keys())
    snapshot_fields = set(df.columns)

    missing = expected_fields - snapshot_fields
    extra = snapshot_fields - expected_fields

    print("\n--- Schema Drift ---")
    print(f"Missing fields: {sorted(missing)}")
    print(f"Extra fields: {sorted(extra)}")

    # --------------------------------------------------------
    # 2. NaN distribution
    # --------------------------------------------------------
    print("\n--- Columns with NaN values ---")
    nan_cols = df.columns[df.isna().any()].tolist()
    for col in nan_cols:
        pct = df[col].isna().mean() * 100
        print(f"{col}: {pct:.2f}% NaN")

    # --------------------------------------------------------
    # 3. Row-level validation
    # --------------------------------------------------------
    print("\n--- Row-Level Validation ---")
    errors = 0

    for idx, row in df.iterrows():
        try:
            FeatureRow(**row.to_dict())
        except Exception as e:
            print(f"Row {idx}: {e}")
            errors += 1
            if errors >= max_errors:
                print(f"Stopping after {max_errors} errors...")
                break

    if errors == 0:
        print("\nAll rows validated successfully.")
    else:
        print(f"\nValidation failed for {errors} rows (first {max_errors} shown).")

    print("\n=== DONE ===")


if __name__ == "__main__":
    validate_features()