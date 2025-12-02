import os
import pandas as pd
import logging
import numpy as np
import argparse

from scripts.fetch_player_stats import main as fetch_stats
from scripts.generate_today_predictions import generate_today_predictions
from scripts.generate_picks import main as generate_picks
from scripts.simulate_bankroll import simulate_bankroll
from config import PREDICTIONS_FILE, PICKS_FILE, PICKS_BANKROLL_FILE

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
        generate_picks(preds_file=PREDICTIONS_FILE, out_file=PICKS_FILE)
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

        preds_df.to_csv(PICKS_BANKROLL_FILE, index=False)
        logger.info("✅ Bankroll simulation completed")