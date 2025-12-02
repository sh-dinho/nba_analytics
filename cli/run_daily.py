# File: cli/run_daily.py

import argparse
from pipelines.daily_pipeline import run_daily_pipeline
from core.logging import setup_logger
import os
import pandas as pd

logger = setup_logger("cli")

def main():
    parser = argparse.ArgumentParser(description="Run NBA daily pipeline")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--strategy", choices=["kelly","flat"], default="kelly")
    parser.add_argument("--max_fraction", type=float, default=0.05)
    parser.add_argument("--export", type=str, default="results/picks.csv")
    # FIX 1: Added --use-synthetic argument
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic player data instead of API fetch") 
    args = parser.parse_args()

    picks_df, metrics = run_daily_pipeline(
        threshold=args.threshold,
        strategy=args.strategy,
        max_fraction=args.max_fraction,
        # FIX 2: Passed the argument value down
        use_synthetic=args.use_synthetic 
    )

    os.makedirs(os.path.dirname(args.export), exist_ok=True)
    picks_df.to_csv(args.export, index=False)
    logger.info(f"Picks exported to {args.export}")
    logger.info(f"Bankroll metrics: {metrics}")

if __name__ == "__main__":
    main()