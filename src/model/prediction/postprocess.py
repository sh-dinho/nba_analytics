from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Prediction Postprocessing
# File: src/model/prediction/postprocess.py
# Author: Sadiq
# ============================================================

import pandas as pd


def apply_threshold(
    df: pd.DataFrame,
    threshold: float = 0.5,
    prob_col: str = "win_probability",
    out_col: str = "predicted_win",
) -> pd.DataFrame:
    """
    Apply a classification threshold to a probability column.

    Enhancements:
        - Validates probability column exists
        - Validates probability column is numeric
        - Validates threshold is in [0, 1]
        - Clips probabilities for safety
    """

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------
    if prob_col not in df.columns:
        raise ValueError(f"apply_threshold: '{prob_col}' column missing.")

    if not pd.api.types.is_numeric_dtype(df[prob_col]):
        raise TypeError(
            f"apply_threshold: '{prob_col}' must be numeric, "
            f"got dtype={df[prob_col].dtype}"
        )

    if not (0.0 <= threshold <= 1.0):
        raise ValueError("apply_threshold: threshold must be between 0 and 1.")

    # --------------------------------------------------------
    # Apply threshold
    # --------------------------------------------------------
    out = df.copy()

    # Clip probabilities for numerical safety
    probs = out[prob_col].clip(0.0, 1.0)

    out[out_col] = (probs >= threshold).astype(int)

    return out
