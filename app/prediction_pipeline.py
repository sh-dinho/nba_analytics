# File: prediction_pipeline.py

import pandas as pd
from scripts.fetch_player_stats import main as fetch_stats
from scripts.generate_today_predictions import generate_today_predictions
from scripts.generate_picks import main as generate_picks
from scripts.simulate_bankroll import simulate_bankroll

class PredictionPipeline:
    def __init__(self, threshold=0.6, strategy="kelly", max_fraction=0.05, results_dir="results", use_synthetic=False):
        self.threshold = threshold
        self.strategy = strategy
        self.max_fraction = max_fraction
        self.results_dir = results_dir
        self.use_synthetic = use_synthetic # <-- NEW ATTRIBUTE ADDED HERE

    def run(self):
        # 1️⃣ Fetch player stats
        # Pass the new argument down to handle synthetic data or live fetch
        fetch_stats(use_synthetic=self.use_synthetic) 

        # 2️⃣ Generate predictions
        preds_df = generate_today_predictions(threshold=self.threshold)

        if preds_df.empty:
            print("No predictions available today.")
            return None, None

        # 3️⃣ Generate picks
        generate_picks(preds_file=f"{self.results_dir}/predictions.csv",
                       out_file=f"{self.results_dir}/picks.csv")

        # 4️⃣ Simulate bankroll
        sim_df = preds_df.rename(columns={"pred_home_win_prob": "prob"})
        history, metrics = simulate_bankroll(sim_df[["decimal_odds", "prob", "ev"]],
                                             strategy=self.strategy,
                                             max_fraction=self.max_fraction)

        # Save final bankroll trajectory
        preds_df["bankroll"] = history[1:][:len(preds_df)]
        preds_df.to_csv(f"{self.results_dir}/picks_bankroll.csv", index=False)
        return preds_df, metrics