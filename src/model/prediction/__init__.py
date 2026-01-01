"""
Prediction module exports.

This file exposes the public prediction API:
    - predict_moneyline
    - predict_totals
    - predict_spread
    - apply_threshold
"""

from .predict import (
    predict_moneyline,
    predict_totals,
    predict_spread,
    apply_threshold,
)

__all__: list[str] = [
    "predict_moneyline",
    "predict_totals",
    "predict_spread",
    "apply_threshold",
]