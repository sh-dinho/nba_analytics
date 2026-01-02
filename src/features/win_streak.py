from __future__ import annotations
import pandas as pd

def add_win_streak(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["team", "date"]).copy()

    shifted = out.groupby("team")["win"].shift(1)
    streak_groups = (shifted != 1).cumsum()
    out["win_streak"] = shifted.groupby(streak_groups).cumcount()

    return out