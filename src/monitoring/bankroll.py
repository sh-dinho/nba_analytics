# ============================================================
# File: src/monitoring/bankroll.py
# ============================================================

import pandas as pd


def simulate_bankroll(bet_log: pd.DataFrame, starting_bankroll: float = 100.0):
    """
    Simulate bankroll growth over time.
    bet_log must contain: date, stake, decimal_odds, actual_result
    """
    df = bet_log.copy().sort_values("date")

    bankroll = starting_bankroll
    bankroll_history = []

    for _, row in df.iterrows():
        stake = row["stake"]
        odds = row["decimal_odds"]
        result = row["actual_result"]

        if result == 1:
            pnl = stake * (odds - 1)
        else:
            pnl = -stake

        bankroll += pnl
        bankroll_history.append(bankroll)

    df["bankroll_after"] = bankroll_history
    return df
