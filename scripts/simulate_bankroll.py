# ============================================================
# File: scripts/simulate_bankroll.py
# Purpose: Simulate bankroll trajectory for daily picks with EV/Kelly, charting, and Telegram-safe notifications
# ============================================================

import pandas as pd
import random
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from nba_core.log_config import init_global_logger
from nba_core.exceptions import DataError, FileError, PipelineError
from nba_core.paths import RESULTS_DIR
from notifications import send_telegram_message, send_photo
from scripts.betting_utils import expected_value, kelly_fraction, american_to_decimal

logger = init_global_logger("simulate_bankroll")

def simulate_bankroll(preds_df: pd.DataFrame,
                      strategy: str = "kelly",
                      max_fraction: float = 0.05,
                      bankroll: float = 1000.0,
                      seed: int | None = None) -> tuple[pd.DataFrame, list[float], dict]:
    if not {"prob", "american_odds"}.issubset(preds_df.columns):
        raise DataError("preds_df must contain 'prob' and 'american_odds' columns")

    if seed is not None:
        random.seed(seed)

    history = []
    current_bankroll = float(bankroll)

    for idx, row in preds_df.iterrows():
        prob, odds = row["prob"], row["american_odds"]
        if pd.isna(prob) or pd.isna(odds):
            preds_df.loc[idx, ["EV", "Kelly_Bet", "bankroll", "outcome"]] = [None, None, current_bankroll, None]
            continue

        try:
            kelly_bet = kelly_fraction(prob, odds, bankroll=current_bankroll, max_fraction=max_fraction)
        except Exception as e:
            raise FileError(f"Kelly calculation failed for odds={odds}, prob={prob}") from e

        preds_df.at[idx, "Kelly_Bet"] = kelly_bet
        bet_size = min(kelly_bet, current_bankroll * max_fraction) if strategy == "kelly" else current_bankroll * max_fraction
        ev = expected_value(prob, odds, stake=bet_size)
        preds_df.at[idx, "EV"] = ev

        outcome = "WIN" if random.random() < prob else "LOSS"
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
        history.append(round(current_bankroll, 2))

        logger.info(
            f"Bet {idx}: prob={prob:.3f}, odds={odds}, EV={ev:.2f}, bet={bet_size:.2f}, outcome={outcome}, bankroll={current_bankroll:.2f}"
        )

    wins = (preds_df["outcome"] == "WIN").sum()
    total_bets = preds_df["outcome"].notna().sum()
    win_rate = wins / total_bets if total_bets > 0 else 0.0

    metrics = {
        "date": pd.Timestamp.today().date().isoformat(),
        "final_bankroll": round(current_bankroll, 2),
        "avg_EV": float(preds_df["EV"].mean(skipna=True)),
        "avg_Kelly_Bet": float(preds_df["Kelly_Bet"].mean(skipna=True)),
        "win_rate": round(win_rate, 4),
        "total_bets": int(total_bets),
    }

    logger.info(
        f"📊 Simulation complete | Final bankroll={metrics['final_bankroll']:.2f}, Avg EV={metrics['avg_EV']:.3f}, "
        f"Avg Kelly Bet={metrics['avg_Kelly_Bet']:.2f}, Win Rate={metrics['win_rate']:.2%}, Total bets={metrics['total_bets']}"
    )
    return preds_df, history, metrics

def plot_trajectory(history: list[float], chart_path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.title("Daily Bankroll Trajectory")
    plt.xlabel("Bet Number")
    plt.ylabel("Bankroll")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(chart_path)
    logger.info(f"📈 Bankroll chart saved to {chart_path}")
    return chart_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate daily bankroll trajectory")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="kelly", choices=["kelly", "flat"])
    parser.add_argument("--max_fraction", type=float, default=0.05)
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
        enriched, history, metrics = simulate_bankroll(df, args.strategy, args.max_fraction, args.bankroll, args.seed)

        today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
        daily_csv = RESULTS_DIR / f"dashboard/bankroll_{today_str}.csv"
        daily_csv.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(daily_csv, index=False)
        chart_path = RESULTS_DIR / f"dashboard/bankroll_{today_str}.png"
        if history:
            plot_trajectory(history, chart_path)

        if args.notify:
            msg = (
                f"🏀 Daily Bankroll Simulation ({today_str})\n"
                f"💰 Final Bankroll: {metrics['final_bankroll']:.2f}\n"
                f"📈 Win Rate: {metrics['win_rate']:.2%}\n"
                f"📊 Total Bets: {metrics['total_bets']}\n"
                f"💵 Avg EV: {metrics['avg_EV']:.3f}\n"
                f"🎯 Avg Kelly Bet: {metrics['avg_Kelly_Bet']:.2f}"
            )
            try:
                send_telegram_message(msg)
                if history:
                    send_photo(str(chart_path), caption="📈 Daily Bankroll Trajectory")
            except Exception as e:
                logger.warning(f"⚠️ Telegram notification failed: {e}")
    except Exception as e:
        logger.error(f"❌ Daily simulation failed: {e}")
        raise PipelineError(f"Daily simulation failed: {e}")
