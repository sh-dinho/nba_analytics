# File: prediction_pipeline.py

import os
import pandas as pd
import logging
import numpy as np
import argparse

from scripts.fetch_player_stats import main as fetch_stats
from scripts.generate_today_predictions import generate_today_predictions
from scripts.generate_picks import main as generate_picks
from scripts.simulate_bankroll import simulate_bankroll

logger = logging.getLogger("PredictionPipeline")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

class PredictionPipeline:
    def __init__(self, threshold=0.6, strategy="kelly", max_fraction=0.05,
                 results_dir="results", use_synthetic=False):
        self.threshold = threshold
        self.strategy = strategy
        self.max_fraction = max_fraction
        self.results_dir = results_dir
        self.use_synthetic = use_synthetic

    def run(self):
        os.makedirs(self.results_dir, exist_ok=True)

        # 1️⃣ Fetch player stats
        fetch_stats(use_synthetic=self.use_synthetic)
        logger.info("✅ Player stats fetched")

        # 2️⃣ Generate predictions
        preds_df = generate_today_predictions(threshold=self.threshold)
        if preds_df.empty:
            logger.warning("No predictions available today.")
            return None, None
        logger.info("✅ Predictions generated")

        # 3️⃣ Generate picks
        generate_picks(
            preds_file=f"{self.results_dir}/predictions.csv",
            out_file=f"{self.results_dir}/picks.csv"
        )
        logger.info("✅ Picks generated")

        # 4️⃣ Simulate bankroll
        sim_df = preds_df.rename(columns={"pred_home_win_prob": "prob"})
        history, metrics = simulate_bankroll(
            sim_df[["decimal_odds", "prob", "ev"]],
            strategy=self.strategy,
            max_fraction=self.max_fraction
        )

        if len(history) > 0:
            avg_traj = np.mean(history, axis=0)
            preds_df["bankroll"] = avg_traj[:len(preds_df)]
        else:
            preds_df["bankroll"] = 0

        preds_df.to_csv(f"{self.results_dir}/picks_bankroll.csv", index=False)
        logger.info("✅ Bankroll simulation complete")

        return preds_df, metrics


# CLI entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the NBA prediction pipeline")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Prediction threshold for picks")
    parser.add_argument("--strategy", type=str, default="kelly",
                        help="Bankroll strategy (e.g., kelly, flat)")
    parser.add_argument("--max_fraction", type=float, default=0.05,
                        help="Maximum fraction of bankroll to bet")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use synthetic stats fallback instead of live scraping")

    args = parser.parse_args()

    pipeline = PredictionPipeline(
        threshold=args.threshold,
        strategy=args.strategy,
        max_fraction=args.max_fraction,
        results_dir=args.results_dir,
        use_synthetic=args.use_synthetic
    )

    preds_df, metrics = pipeline.run()
    if preds_df is not None:
        logger.info("✅ Pipeline finished successfully")
        logger.info(f"Metrics: {metrics}")