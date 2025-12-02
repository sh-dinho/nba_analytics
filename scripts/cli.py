# scripts/cli.py
import argparse
from scripts.simulate_bankroll import simulate_bankroll
from scripts.generate_today_predictions import generate_today_predictions
from scripts.generate_picks import main as generate_picks
from scripts.utils import setup_logger
import pandas as pd

logger = setup_logger("cli")

def run_pipeline(threshold, strategy, max_fraction, export="results/picks.csv"):
    preds_df = generate_today_predictions(threshold=threshold)
    picks_df = generate_picks()
    history, metrics = simulate_bankroll(preds_df, strategy=strategy, max_fraction=max_fraction)

    picks_df["bankroll"] = history[:len(picks_df)]
    picks_df.to_csv(export, index=False)
    logger.info(f"Picks and bankroll saved to {export}")
    logger.info(f"Bankroll metrics: {metrics}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--strategy", choices=["kelly","flat"], default="kelly")
    parser.add_argument("--max_fraction", type=float, default=0.05)
    parser.add_argument("--export", type=str, default="results/picks.csv")
    args = parser.parse_args()

    run_pipeline(args.threshold, args.strategy, args.max_fraction, args.export)

if __name__ == "__main__":
    main()
