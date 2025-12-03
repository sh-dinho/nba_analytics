# ============================================================
# File: scripts/simulate_bankroll.py
# Purpose: Simulate bankroll trajectory with EV and Kelly bet sizes
# ============================================================

import pandas as pd
from scripts.betting_utils import expected_value, calculate_kelly_criterion
from core.log_config import setup_logger
from core.exceptions import DataError

logger = setup_logger("simulate_bankroll")


def simulate_bankroll(preds_df: pd.DataFrame,
                      strategy: str = "kelly",
                      max_fraction: float = 0.05,
                      bankroll: float = 1000.0):
    """
    Simulates bankroll evolution given predictions and odds.
    Adds EV and Kelly bet size columns to preds_df.

    Args:
        preds_df: DataFrame containing predictions with 'prob' and 'american_odds' columns.
        strategy: Betting strategy ("kelly" or "flat").
        max_fraction: Maximum fraction of bankroll to risk per bet.
        bankroll: Starting bankroll.

    Returns:
        history: List of bankroll values after each bet.
        metrics: Dict with final bankroll, average EV, and average Kelly bet size.
    """
    if not {"prob", "american_odds"}.issubset(preds_df.columns):
        raise DataError("preds_df must contain 'prob' and 'american_odds' columns")

    history = []
    current_bankroll = bankroll

    for idx, row in preds_df.iterrows():
        prob = row.get("prob")
        odds = row.get("american_odds")

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
            profit = (ev / 100.0) * bet_size
            current_bankroll += profit
            outcome = "WIN"
        else:
            current_bankroll -= bet_size
            outcome = "LOSS"

        preds_df.at[idx, "bankroll"] = current_bankroll
        preds_df.at[idx, "outcome"] = outcome
        history.append(current_bankroll)

        logger.info(f"Game {idx}: prob={prob:.3f}, odds={odds}, EV={ev:.2f}, "
                    f"bet={bet_size:.2f}, outcome={outcome}, bankroll={current_bankroll:.2f}")

    metrics = {
        "final_bankroll": current_bankroll,
        "avg_EV": preds_df["EV"].mean(skipna=True),
        "avg_Kelly_Bet": preds_df["Kelly_Bet"].mean(skipna=True),
    }

    logger.info(f"Simulation completed | Final bankroll={metrics['final_bankroll']:.2f}, "
                f"Avg EV={metrics['avg_EV']:.3f}, Avg Kelly Bet={metrics['avg_Kelly_Bet']:.2f}")

    return history, metrics