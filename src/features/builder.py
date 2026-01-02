from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Feature Builder (Version‑Agnostic)
# File: src/features/builder.py
# Author: Sadiq
# ============================================================

import pandas as pd
from loguru import logger

from src.features.feature_pipeline import build_features
from src.features.feature_schema import FeatureRow


class FeatureBuilder:
    """
    Version‑agnostic feature builder.
    Wraps the canonical feature pipeline and exposes a stable API.
    """

    def __init__(self, version: str | None = None):
        self.version = version

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def build(self, long_df: pd.DataFrame, persist: bool = False) -> pd.DataFrame:
        """
        Build full feature matrix from canonical long-format input.
        """
        logger.info(f"🏗️  FeatureBuilder: building features for {len(long_df)} rows...")
        return build_features(long_df, persist=persist)

    # ------------------------------------------------------------
    # Canonical expected columns
    # ------------------------------------------------------------
    def expected_feature_columns(self) -> list[str]:
        """
        Returns the canonical list of feature columns defined by FeatureRow.
        This is the single source of truth for schema validation.
        """
        return list(FeatureRow.model_fields.keys())

    # ------------------------------------------------------------
    # Optional: convenience helper for schema drift checks
    # ------------------------------------------------------------
    def validate_columns(self, df: pd.DataFrame) -> dict:
        """
        Compare DataFrame columns against canonical FeatureRow schema.
        Returns a dict with 'missing' and 'extra' keys.
        """
        expected = set(self.expected_feature_columns())
        actual = set(df.columns)

        return {
            "missing": expected - actual,
            "extra": actual - expected,
        }