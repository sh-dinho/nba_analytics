# ============================================================
# File: scripts/simulate_bankroll.py
# Purpose: Simulate bankroll trajectory with EV and Kelly bet sizes
# ============================================================

import pandas as pd
import random
from scripts.betting_utils import expected_value, calculate_kelly_criterion, american_to_decimal
from core.log_config import setup_logger
from core.exceptions import DataError

logger = setup_logger("simulate_bankroll")


def simulate_bankroll(preds_df: pd.DataFrame,
                      strategy: str = "kelly",
                      max_fraction: float = 0.05,
                      bankroll: float = 1000.0,
                      seed: int | None = None,
                      output_file: str | None = None):
    """
    Simulates bankroll evolution given predictions and odds.
    Adds EV, Kelly bet size, bankroll trajectory, and outcome columns to preds_df.

    Args:
        preds_df: DataFrame containing predictions with 'prob' and 'american_odds' columns.
        strategy: Betting strategy ("kelly" or "flat").
        max_fraction: Maximum fraction of bankroll to risk per bet.
        bankroll: Starting bankroll.
        seed: Optional random seed for reproducibility.
        output_file: Optional path to save enriched DataFrame.

    Returns:
        preds_df: Enriched DataFrame with EV, Kelly_Bet, bankroll, outcome columns.
        history: List of bankroll values after each bet.
        metrics: Dict with final bankroll, average EV, average Kelly bet size, win rate.
    """
    if not {"prob", "american_odds"}.issubset(preds_df.columns):
        raise DataError("preds_df must contain 'prob' and 'american_odds' columns")

    if seed is not None:
        random.seed(seed)

    history = []
    current_bankroll = bankroll

    for idx, row in preds_df.iterrows():
        prob, odds = row["prob"], row["american_odds"]

        if pd.isna(prob) or pd.isna(odds):
            preds_df.at[idx, "EV"] = None
            preds_df.at[idx, "Kelly_Bet"] = None
            preds_df.at[idx, "bankroll"] = current_bankroll
            preds_df.at[idx, "outcome"] = None
            continue

        # Kelly bet size in dollars
        kelly_bet = calculate_kelly_criterion(odds, prob, current_bankroll)
        preds_df.at[idx, "Kelly_Bet"] = kelly_bet

        # Apply strategy
        if strategy == "kelly":
            bet_size = min(kelly_bet, current_bankroll * max_fraction)
        else:  # flat betting
            bet_size = current_bankroll * max_fraction

        # Expected value with actual bet size
        ev = expected_value(prob, odds, stake=bet_size)
        preds_df.at[idx, "EV"] = ev

        # Simulate outcome realistically (Bernoulli trial)
        outcome = "WIN" if random.random() < prob else "LOSS"

        # Profit calculation using decimal odds
        try:
            dec_odds = american_to_decimal(odds)
        except DataError as e:
            logger.error(f"Invalid odds {odds}: {e}")
            preds_df.at[idx, "bankroll"] = current_bankroll
            preds_df.at[idx, "outcome"] = None
            continue

        profit = bet_size * (dec_odds - 1) if outcome == "WIN" else -bet_size
        current_bankroll += profit

        preds_df.at[idx, "bankroll"] = current_bankroll
        preds_df.at[idx, "outcome"] = outcome
        history.append(current_bankroll)

        logger.info(
            f"Game {idx}: prob={prob:.3f}, odds={odds}, EV={ev:.2f}, "
            f"bet={bet_size:.2f}, outcome={outcome}, bankroll={current_bankroll:.2f}"
        )

    wins = (preds_df["outcome"] == "WIN").sum()
    total_bets = preds_df["outcome"].notna().sum()
    win_rate = wins / total_bets if total_bets > 0 else 0

    metrics = {
        "final_bankroll": current_bankroll,
        "avg_EV": preds_df["EV"].mean(skipna=True),
        "avg_Kelly_Bet": preds_df["Kelly_Bet"].mean(skipna=True),
        "win_rate": win_rate,
        "total_bets": total_bets,
    }

    logger.info(
        f"Simulation completed | Final bankroll={metrics['final_bankroll']:.2f}, "
        f"Avg EV={metrics['avg_EV']:.3f}, Avg Kelly Bet={metrics['avg_Kelly_Bet']:.2f}, "
        f"Win Rate={metrics['win_rate']:.2%}"
    )

    if output_file:
        preds_df.to_csv(output_file, index=False)
        logger.info(f"📑 Simulation results saved to {output_file}")

    return preds_df, history, metrics