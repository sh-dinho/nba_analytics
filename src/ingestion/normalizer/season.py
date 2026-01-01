from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Season Inference
# File: src/ingestion/normalizer/season.py
# Author: Sadiq
#
# Description:
#     Canonical NBA season inference logic for ingestion.
#     Produces labels like "2024-25" based on game date.
# ============================================================

from datetime import date, datetime


def infer_season_label(d: date) -> str:
    """
    Canonical NBA season inference:
      - If month >= 10 → season starts this year
      - Else → season started last year

    Returns:
        A season label like "2024-25".
    """
    if isinstance(d, datetime):
        d = d.date()

    start_year = d.year if d.month >= 10 else d.year - 1
    end_suffix = str(start_year + 1)[-2:]
    return f"{start_year}-{end_suffix}"