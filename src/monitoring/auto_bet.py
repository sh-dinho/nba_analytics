# ============================================================
# File: src/monitoring/auto_bet.py
# ============================================================

import pandas as pd
from src.monitoring.bet_logger import log_recommended_bets


def auto_bet(bets_df: pd.DataFrame, bankroll=100, kelly_cap=0.05):
    """
    Automatically place bets based on Kelly and log them.
    bets_df must contain: game_id, market_team, model_prob, implied_prob, edge,
                          decimal_odds, kelly_fraction
    """
    df = bets_df.copy()
    df["stake"] = bankroll * df["kelly_fraction"].clip(upper=kelly_cap)

    to_log = df.rename(columns={"market_team": "team"})
    to_log["date"] = pd.Timestamp.today().date().isoformat()

    log_recommended_bets(
        to_log[
            [
                "game_id",
                "date",
                "team",
                "model_prob",
                "implied_prob",
                "edge",
                "decimal_odds",
                "stake",
            ]
        ]
    )

    return df
