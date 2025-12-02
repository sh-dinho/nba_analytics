# ============================================================
# File: scripts/simulate_bankroll.py
# Purpose: Simulate bankroll trajectory with EV and Kelly bet sizes
# ============================================================

import pandas as pd
from scripts.betting_utils import expected_value, calculate_kelly_criterion

def simulate_bankroll(preds_df, strategy="kelly", max_fraction=0.05, bankroll=1000.0):
    """
    Simulates bankroll evolution given predictions and odds.
    Adds EV and Kelly bet size columns to preds_df.
    """
    history = []
    current_bankroll = bankroll

    for idx, row in preds_df.iterrows():
        prob = row.get("prob", None)
        odds = row.get("american_odds", None)

        if prob is None or odds is None:
            preds_df.at[idx, "EV"] = None
            preds_df.at[idx, "Kelly_Bet"] = None
            continue

        # Expected value for $100 stake
        ev = expected_value(prob, odds, stake=100)
        preds_df.at[idx, "EV"] = ev

        # Kelly bet size in dollars
        kelly_bet = calculate_kelly_criterion(odds, prob, current_bankroll)
        preds_df.at[idx, "Kelly_Bet"] = kelly_bet

        # Apply strategy
        if strategy == "kelly":
            bet_size = min(kelly_bet, current_bankroll * max_fraction)
        else:
            bet_size = current_bankroll * max_fraction

        # Simulate outcome (placeholder: assume win if prob > 0.5)
        if prob > 0.5:
            profit = (preds_df.at[idx, "EV"] / 100) * bet_size
            current_bankroll += profit
        else:
            current_bankroll -= bet_size

        preds_df.at[idx, "bankroll"] = current_bankroll
        history.append(current_bankroll)

    metrics = {
        "final_bankroll": current_bankroll,
        "avg_EV": preds_df["EV"].mean(),
        "avg_Kelly_Bet": preds_df["Kelly_Bet"].mean()
    }

    return history, metrics