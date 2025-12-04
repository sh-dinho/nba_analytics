# ============================================================
# File: scripts/simulate_bankroll.py
# Purpose: Simulate bankroll trajectory with EV and Kelly bet sizes + Telegram notifications
# ============================================================

import pandas as pd
import random
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from scripts.betting_utils import expected_value, kelly_fraction, american_to_decimal
from core.log_config import init_global_logger
from core.exceptions import DataError, FileError, PipelineError
from core.paths import RESULTS_DIR
from notifications import send_telegram_message, send_photo  # ✅ Telegram hooks

logger = init_global_logger()

SIM_SUMMARY_FILE = RESULTS_DIR / "bankroll_simulation_summary.csv"
SIM_WEEKLY_FILE = RESULTS_DIR / "bankroll_simulation_weekly.csv"
SIM_MONTHLY_FILE = RESULTS_DIR / "bankroll_simulation_monthly.csv"


# === Simulation Core ===

def simulate_bankroll(
    preds_df: pd.DataFrame,
    strategy: str = "kelly",
    max_fraction: float = 0.05,
    bankroll: float = 1000.0,
    seed: int | None = None,
    output_file: str | None = None,
):
    """Simulates bankroll evolution given predictions and odds."""
    if not {"prob", "american_odds"}.issubset(preds_df.columns):
        raise DataError("preds_df must contain 'prob' and 'american_odds' columns")

    if seed is not None:
        random.seed(seed)

    history: list[float] = []
    current_bankroll = float(bankroll)

    for idx, row in preds_df.iterrows():
        prob, odds = row["prob"], row["american_odds"]

        if pd.isna(prob) or pd.isna(odds):
            preds_df.at[idx, "EV"] = None
            preds_df.at[idx, "Kelly_Bet"] = None
            preds_df.at[idx, "bankroll"] = current_bankroll
            preds_df.at[idx, "outcome"] = None
            continue

        try:
            kelly_bet = kelly_fraction(prob, odds, bankroll=current_bankroll, max_fraction=max_fraction)
        except Exception as e:
            raise FileError(f"Kelly calculation failed for odds={odds}, prob={prob}", file_path=str(output_file)) from e

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
            f"Game {idx}: prob={prob:.3f}, odds={odds}, EV={ev:.2f}, "
            f"bet={bet_size:.2f}, outcome={outcome}, bankroll={current_bankroll:.2f}"
        )

    wins = (preds_df["outcome"] == "WIN").sum()
    total_bets = preds_df["outcome"].notna().sum()
    win_rate = wins / total_bets if total_bets > 0 else 0.0

    metrics = {
        "Date": pd.Timestamp.today().date().isoformat(),
        "final_bankroll": round(current_bankroll, 2),
        "avg_EV": float(preds_df["EV"].mean(skipna=True)),
        "avg_Kelly_Bet": float(preds_df["Kelly_Bet"].mean(skipna=True)),
        "win_rate": round(win_rate, 4),
        "total_bets": int(total_bets),
    }

    logger.info(
        f"📊 Simulation completed | Final bankroll={metrics['final_bankroll']:.2f}, "
        f"Avg EV={metrics['avg_EV']:.3f}, Avg Kelly Bet={metrics['avg_Kelly_Bet']:.2f}, "
        f"Win Rate={metrics['win_rate']:.2%}, Total bets={metrics['total_bets']}"
    )

    if output_file:
        try:
            preds_df.to_csv(output_file, index=False)
            logger.info(f"📑 Simulation results saved to {output_file}")
        except Exception as e:
            raise FileError(f"Failed to save simulation results to {output_file}", file_path=str(output_file)) from e

    return preds_df, history, metrics


# === Plotting ===

def plot_trajectory(history: list[float], chart_path: Path):
    """Plot bankroll trajectory and save chart."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.title("Bankroll Trajectory")
    plt.xlabel("Bet Number")
    plt.ylabel("Bankroll")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(chart_path)
    logger.info(f"📈 Bankroll chart saved to {chart_path}")
    return chart_path


# === Summaries ===

def export_simulation_summary(metrics: dict):
    """Append simulation metrics to bankroll_simulation_summary.csv."""
    df = pd.DataFrame([metrics])
    if SIM_SUMMARY_FILE.exists():
        df.to_csv(SIM_SUMMARY_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(SIM_SUMMARY_FILE, index=False)
    logger.info(f"📑 Simulation summary appended to {SIM_SUMMARY_FILE}")


def log_weekly_summary():
    """Aggregate simulation bankroll changes by week."""
    if not SIM_SUMMARY_FILE.exists():
        return
    df = pd.read_csv(SIM_SUMMARY_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Week"] = df["Date"].dt.to_period("W").astype(str)
    weekly = df.groupby("Week").agg({
        "final_bankroll": "last",
        "avg_EV": "mean",
        "avg_Kelly_Bet": "mean",
        "win_rate": "mean",
        "total_bets": "sum"
    }).reset_index()
    weekly.to_csv(SIM_WEEKLY_FILE, index=False)
    logger.info(f"📑 Weekly simulation summary exported to {SIM_WEEKLY_FILE}")


def log_monthly_summary():
    """Aggregate simulation bankroll changes by month."""
    if not SIM_SUMMARY_FILE.exists():
        return
    df = pd.read_csv(SIM_SUMMARY_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    monthly = df.groupby("Month").agg({
        "final_bankroll": "last",
        "avg_EV": "mean",
        "avg_Kelly_Bet": "mean",
        "win_rate": "mean",
        "total_bets": "sum"
    }).reset_index()
    monthly.to_csv(SIM_MONTHLY_FILE, index=False)
    logger.info(f"📑 Monthly simulation summary exported to {SIM_MONTHLY_FILE}")

def log_weekly_summary():
    """Aggregate simulation bankroll changes by week with cumulative bankroll."""
    if not SIM_SUMMARY_FILE.exists():
        return
    df = pd.read_csv(SIM_SUMMARY_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Week"] = df["Date"].dt.to_period("W").astype(str)

    weekly = df.groupby("Week").agg({
        "final_bankroll": "last",
        "avg_EV": "mean",
        "avg_Kelly_Bet": "mean",
        "win_rate": "mean",
        "total_bets": "sum"
    }).reset_index()

    # Add cumulative bankroll progression
    weekly["Cumulative_Bankroll"] = weekly["final_bankroll"].cummax()

    weekly.to_csv(SIM_WEEKLY_FILE, index=False)
    logger.info(f"📑 Weekly simulation summary exported to {SIM_WEEKLY_FILE}")


def log_monthly_summary():
    """Aggregate simulation bankroll changes by month with cumulative bankroll."""
    if not SIM_SUMMARY_FILE.exists():
        return
    df = pd.read_csv(SIM_SUMMARY_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    monthly = df.groupby("Month").agg({
        "final_bankroll": "last",
        "avg_EV": "mean",
        "avg_Kelly_Bet": "mean",
        "win_rate": "mean",
        "total_bets": "sum"
    }).reset_index()

    # Add cumulative bankroll progression
    monthly["Cumulative_Bankroll"] = monthly["final_bankroll"].cummax()

    monthly.to_csv(SIM_MONTHLY_FILE, index=False)
    logger.info(f"📑 Monthly simulation summary exported to {SIM_MONTHLY_FILE}")


def plot_weekly_summary():
    """Generate a line chart of cumulative bankroll progression by week."""
    if not SIM_WEEKLY_FILE.exists():
        logger.warning("⚠️ No weekly summary file found.")
        return None
    df = pd.read_csv(SIM_WEEKLY_FILE)
    if df.empty:
        return None

    chart_path = RESULTS_DIR / "weekly_bankroll_chart.png"
    plt.figure(figsize=(10, 6))
    plt.plot(df["Week"], df["Cumulative_Bankroll"], marker="o", label="Cumulative Bankroll")
    plt.title("Weekly Bankroll Progression")
    plt.xlabel("Week")
    plt.ylabel("Cumulative Bankroll")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path)
    logger.info(f"📈 Weekly bankroll chart saved to {chart_path}")
    return chart_path


def plot_monthly_summary():
    """Generate a line chart of cumulative bankroll progression by month."""
    if not SIM_MONTHLY_FILE.exists():
        logger.warning("⚠️ No monthly summary file found.")
        return None
    df = pd.read_csv(SIM_MONTHLY_FILE)
    if df.empty:
        return None

    chart_path = RESULTS_DIR / "monthly_bankroll_chart.png"
    plt.figure(figsize=(10, 6))
    plt.plot(df["Month"], df["Cumulative_Bankroll"], marker="s", color="green", label="Cumulative Bankroll")
    plt.title("Monthly Bankroll Progression")
    plt.xlabel("Month")
    plt.ylabel("Cumulative Bankroll")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path)
    logger.info(f"📈 Monthly bankroll chart saved to {chart_path}")
    return chart_path

# === CLI ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate bankroll trajectory with EV and Kelly bet sizes")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to CSV file containing predictions with 'prob' and 'american_odds' columns")
    parser.add_argument("--strategy", type=str, default="kelly", choices=["kelly", "flat"],
                        help="Betting strategy: 'kelly' or 'flat'")
    parser.add_argument("--max_fraction", type=float, default=0.05,
                        help="Maximum fraction of bankroll to risk per bet")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Starting bankroll")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save enriched DataFrame with simulation results")
    parser.add_argument("--export-json", type=str, default=None,
                        help="Optional path to export simulation results as JSON")
    parser.add_argument("--chart", type=str, default=None,
                        help="Optional path to save bankroll trajectory chart")
    parser.add_argument("--notify", action="store_true",
                        help="Send final metrics and chart to Telegram")
    parser.add_argument("--export-summary", action="store_true",
                        help="Append metrics to bankroll_simulation_summary.csv and update weekly/monthly summaries")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
        enriched, history, metrics = simulate_bankroll(
            preds_df=df,
            strategy=args.strategy,
            max_fraction=args.max_fraction,
            bankroll=args.bankroll,
            seed=args.seed,
            output_file=args.output,
        )

        if args.export_json:
            enriched.to_json(args.export_json, orient="records", indent=2)
            logger.info(f"📑 Simulation results also exported to JSON at {args.export_json}")

        chart_path = None
        if args.chart and history:
            chart_path = plot_trajectory(history, Path(args.chart))

        if args.export_summary:
            export_simulation_summary(metrics)
            log_weekly_summary()
            log_monthly_summary()

        if args.notify:
            msg = (
                f"🏀 Bankroll Simulation Complete\n"
                f"💰 Final Bankroll: {metrics['final_bankroll']:.2f}\n"
                f"📈 Win Rate: {metrics['win_rate']:.2%}\n"
                f"📊 Total Bets: {metrics['total_bets']}\n"
                f"💵 Avg EV: {metrics['avg_EV']:.3f}\n"
                f"🎯 Avg Kelly Bet: {metrics['avg_Kelly_Bet']:.2f}"
            )
            send_telegram_message(msg)
            if chart_path:
                send_photo(str(chart_path), caption="📈 Bankroll Trajectory")

    except Exception as e:
        logger.error(f"❌ Simulation failed: {e}")
        raise PipelineError(f"Simulation failed: {e}")