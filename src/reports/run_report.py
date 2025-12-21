from __future__ import annotations

import argparse
from loguru import logger

from src.backtest.engine import BacktestConfig
from src.reports.backtest_report import generate_backtest_accuracy_report


def main():
    parser = argparse.ArgumentParser(description="Generate backtest + accuracy report")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--min_edge", type=float, default=0.03)
    parser.add_argument("--kelly", type=float, default=0.25)
    parser.add_argument("--max_stake", type=float, default=0.05)
    args = parser.parse_args()

    cfg = BacktestConfig(
        starting_bankroll=args.bankroll,
        min_edge=args.min_edge,
        kelly_fraction=args.kelly,
        max_stake_fraction=args.max_stake,
    )

    logger.info("Generating report...")
    path = generate_backtest_accuracy_report(
        start_date=args.start,
        end_date=args.end,
        config=cfg,
        decision_threshold=0.5,
    )

    logger.success(f"Report saved to: {path}")


if __name__ == "__main__":
    main()
