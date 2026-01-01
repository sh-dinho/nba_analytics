from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Features Package
# File: src/features/__init__.py
# Author: Sadiq
#
# Description:
#     Public API for feature engineering.
#     Exposes the unified FeatureBuilder and feature pipeline.
# ============================================================

from src.features.builder import FeatureBuilder
from src.features.feature_pipeline import build_features

# Optional: expose feature functions for direct use
from src.features.rolling import add_rolling_features
from src.features.rest import add_rest_features
from src.features.form import add_form_features
from src.features.elo import add_elo_features
from src.features.elo_rolling import add_elo_rolling_features
from src.features.opponent_adjusted import add_opponent_adjusted_features
from src.features.sos import add_sos_features
from src.features.margin_features import add_margin_features


__all__ = [
    "FeatureBuilder",
    "build_features",
    "add_rolling_features",
    "add_rest_features",
    "add_form_features",
    "add_elo_features",
    "add_elo_rolling_features",
    "add_opponent_adjusted_features",
    "add_sos_features",
    "add_margin_features",
]