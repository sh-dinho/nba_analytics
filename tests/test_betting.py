# ============================================================
# File: tests/test_betting.py
# Purpose: Unit tests for betting recommendations
# Version: 2.0
# Author: Your Team
# Date: December 2025
# ============================================================

import pandas as pd
from src.ranking.bet import generate_betting_recommendations


class Config:
    class Betting:
        threshold = 0.6


def test_betting_recommendations_threshold():
    df = pd.DataFrame(
        {
            "gameId": [1],
            "homeTeam": ["Lakers"],
            "awayTeam": ["Celtics"],
            "predicted_win": [0.75],
        }
    )
    recs = generate_betting_recommendations(df, Config())
    assert not recs.empty
    assert recs.iloc[0]["predicted_win"] >= Config.Betting.threshold
