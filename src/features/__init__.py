# ============================================================
# Path: src/features/__init__.py
# Purpose: Initialize features package
# Version: 1.1
# ============================================================

# Expose key functions at package level
from .feature_engineering import generate_features_for_games

__all__ = [
    "generate_features_for_games",
]
