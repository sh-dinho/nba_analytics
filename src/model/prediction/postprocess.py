from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Prediction Postprocessing
# File: src/model/prediction/postprocess.py
# Author: Sadiq
# ============================================================

import pandas as pd


def apply_threshold(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Apply a win/loss threshold to win_probability.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'win_probability' column.
    threshold : float
        Classification threshold for predicted_win.

    Returns
    -------
    pd.DataFrame
        Copy of df with a new 'predicted_win' column.
    """
    if "win_probability" not in df.columns:
        raise ValueError("apply_threshold: 'win_probability' column missing.")

    out = df.copy()
    out["predicted_win"] = (out["win_probability"] >= threshold).astype(int)
    return out