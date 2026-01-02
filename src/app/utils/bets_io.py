from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Module: Bets IO Utilities
# Purpose: Shared helpers for loading bet logs.
# ============================================================

import pandas as pd
from src.config.paths import DATA_DIR


def load_bet_log() -> pd.DataFrame:
    path = DATA_DIR / "bets" / "bet_log.parquet"
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

    # Normalize date/datetime columns if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "placed_at" in df.columns:
        df["placed_at"] = pd.to_datetime(df["placed_at"], errors="coerce")

    return df
