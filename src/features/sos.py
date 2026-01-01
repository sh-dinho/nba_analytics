from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Strength of Schedule Features
# File: src/features/sos.py
# Author: Sadiq
#
# Description:
#     Simple strength-of-schedule proxies for team-game rows,
#     based on recent opponent scoring. Uses a 10-game rolling
#     window of opponent points allowed.
# ============================================================

import pandas as pd


def add_sos_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds strength-of-schedule (SOS) features based on recent
    opponent scoring.

    Required columns:
        - team
        - date
        - game_id
        - opponent_score
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["team", "date", "game_id"])

    # Simple SOS: rolling 10-game average of opponent scoring
    out["sos"] = (
        out.groupby("team")["opponent_score"]
        .transform(lambda s: s.shift().rolling(10, min_periods=1).mean())
    )

    return out