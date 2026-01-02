from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Backtest Report CLI (Canonical)
# File: src/scripts/run_backtest_report.py
# Author: Sadiq
#
# Description:
#     Generates a canonical backtest report using:
#       • canonical predictions
#       • canonical bankroll engine
#       • canonical accuracy + ROI metrics
#
#     Safe for cron, Airflow, GitHub Actions, systemd.
# ============================================================

import argparse
import sys
from datetime import date, timedelta

import pandas as pd
from loguru import logger

from src.backtest.canonical_engine import (
    CanonicalBacktestConfig,
    run_canonical_backtest,
)
from src.config.paths import PREDICTIONS_DIR, LOGS_DIR


# ------------------------------------------------------------
# Date Parsing
# ------------------------------------------------------------

def parse_date_or_range(start: str | None, end: str | None, days_range: int | None):
    """Supports:
      --start 2024-01-01 --end 2024-03-01
      --range 30   (last 30 days)
    """
    if days_range:
        today = date.today()
        start_date = today - timedelta(days=days_range)
        return start_date.isoformat(), today.isoformat()

    if start and end:
        return start, end

    if start and not end:
        raise ValueError("If --start is provided, --end must also be provided.")

    # Default: last 30 days
    today = date.today()
    start_date = today - timedelta(days=30)
    return start_date.isoformat(), today.isoformat()


# ------------------------------------------------------------
# Prediction Loading
# ------------------------------------------------------------

def load_predictions(start: str, end: str) -> pd.DataFrame:
    """Load canonical predictions between start and end dates."""
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)

    files = sorted(PREDICTIONS_DIR.glob("predictions_*.parquet"))
    dfs: list[pd.DataFrame] = []

    for f in files:
        try:
            d_str = f.stem.split("_")[1]
            d = date.fromisoformat(d_str)
            if start_d <= d <= end_d:
                dfs.append(pd.read_parquet(f))
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Dedupe for safety
    if {"game_id", "team"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["game_id", "team"])

    logger.info(f"Loaded {len(df)} prediction rows from {len(dfs)} files.")
    return df


# ------------------------------------------------------------
# Backtest Runner
# ------------------------------------------------------------

def run_backtest_report(
    start: str,
    end: str,
    bankroll: float,
    min_edge: float,
    kelly: float,
    max_stake: float,
) -> dict:
    logger.info(f"Running canonical backtest from {start} → {end}")

    preds = load_predictions(start, end)
    if preds.empty:
        msg = "No predictions found in the given date range."
        logger.error(msg)
        return {"ok": False, "error": msg}

    cfg = CanonicalBacktestConfig(
        starting_bankroll=bankroll,
        min_edge=min_edge,
        kelly_fraction=kelly,
        max_stake_fraction=max_stake,
    )

    try:
        report = run_canonical_backtest(preds, cfg)
    except Exception as e:
        msg = f"Backtest failed: {e}"
        logger.error(msg)
        return {"ok": False, "error": msg}

    # Save report (scalar metrics)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOGS_DIR / f"backtest_{start}_to_{end}.json"
    out_path.write_text(report.to_json(indent=2))

    logger.success(f"Backtest report saved to {out_path}")

    print("\n=== BACKTEST SUMMARY ===")
    print(f"ROI: {report.roi:.2%}")
    print(f"Accuracy: {report.accuracy:.3f}")
    print(f"Final Bankroll: {report.final_bankroll:.2f}")
    print(f"Total Profit: {report.total_profit:.2f}")
    print(f"Max Drawdown: {report.max_drawdown:.2%}")
    print(f"Bets: {report.n_bets}")
    print("=========================\n")

    return {
        "ok": True,
        "report_path": str(out_path),
        "start_date": start,
        "end_date": end,
        "bankroll": bankroll,
        "min_edge": min_edge,
        "kelly_fraction": kelly,
        "max_stake_fraction": max_stake,
    }


# ------------------------------------------------------------
# CLI Entrypoint
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate canonical backtest report")

    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--range", type=int, default=None)

    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--min_edge", type=float, default=0.03)
    parser.add_argument("--kelly", type=float, default=0.25)
    parser.add_argument("--max_stake", type=float, default=0.05)

    args = parser.parse_args()

    try:
        start, end = parse_date_or_range(args.start, args.end, args.range)
    except Exception as e:
        logger.error(f"Invalid date arguments: {e}")
        sys.exit(1)

    result = run_backtest_report(
        start=start,
        end=end,
        bankroll=args.bankroll,
        min_edge=args.min_edge,
        kelly=args.kelly,
        max_stake=args.max_stake,
    )

    if not result.get("ok"):
        sys.exit(1)

    logger.info("🎉 Backtest completed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
