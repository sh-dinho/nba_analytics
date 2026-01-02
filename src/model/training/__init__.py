"""
Public API exports for model training wrappers.

This module exposes the training entrypoints for:
    - train_moneyline
    - train_spread
    - train_totals

These wrappers use the shared training logic and return:
    (model, y_pred, full_metrics_report)
"""

from .moneyline import train_moneyline
from .spread import train_spread
from .totals import train_totals

__all__ = [
    "train_moneyline",
    "train_spread",
    "train_totals",
]
