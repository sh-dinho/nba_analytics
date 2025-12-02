# ============================================================
# File: app/prediction_pipeline.py
# Purpose: NBA Prediction Pipeline – includes EV, Kelly bet sizes, bankroll trajectory, and summary metrics
# ============================================================

import os
import pandas as pd
import logging
import numpy as np
import argparse

from scripts.fetch_player_stats import main as fetch_stats
from scripts.build_features import build_features, build_features_for_new_games
from scripts.train_model import main as train_model
from scripts.fetch_new_games import main as fetch_new_games
from scripts.generate_today_predictions import generate_today_predictions
from scripts.generate_picks import main as generate_picks
from scripts.simulate_bankroll import simulate_bankroll   # <-- updated simulate_bankroll with EV + Kelly
from scripts.sbr_odds_provider import SbrOddsProvider
from scripts.nn_runner import nn_runner
from scripts.xgb_runner import xgb_runner
from config import PREDICTIONS_FILE, PICKS_FILE, PICKS_BANKROLL_FILE

logger = logging.getLogger("PredictionPipeline")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

class PredictionPipeline:
    def __init__(self, threshold=0.6, strategy="kelly", max_fraction=0.05,
                 results_dir="results", use_synthetic=False,
                 sportsbook="fanduel", model_type="logistic"):
        self.threshold = threshold
        self.strategy = strategy
        self.max_fraction = max_fraction
        self.results_dir = results_dir
        self.use_synthetic = use_synthetic
        self.sportsbook = sportsbook
        self.model_type = model_type

    def run(self):
        os.makedirs(self.results_dir, exist_ok=True)

        # 1️⃣ Fetch player stats
        fetch_stats(use_synthetic=self.use_synthetic)
        logger.info("✅ Player stats fetched")

        # 2️⃣ Build features
        build_features()
        logger.info("✅ Features built")

        # 3️⃣ Train model (only for logistic baseline)
        if self.model_type == "logistic":
            train_model()
            logger.info("✅ Logistic model trained")

        # 4️⃣ Fetch today's games
        if self.use_synthetic:
            fetch_new_games(use_synthetic=True)
            logger.info("✅ Synthetic today's games fetched")
        else:
            odds_provider = SbrOddsProvider(sportsbook=self.sportsbook)
            odds_data = odds_provider.get_odds()
            rows = []
            for matchup, data in odds_data.items():
                home, away = matchup.split(":")
                rows.append({
                    "TEAM_HOME": home,
                    "TEAM_AWAY": away,
                    "AGE": 27,  # placeholder
                    "PTS": 20,
                    "AST": 5,
                    "REB": 7,
                    "GAMES_PLAYED": 15,
                    "american_odds": data[home]["money_line_odds"],
                    "OU": data["under_over_odds"]
                })
            df = pd.DataFrame(rows)
            os.makedirs("data", exist_ok=True)
            df.to_csv("data/new_games.csv", index=False)
            logger.info("✅ Real today's games with odds saved to data/new_games.csv")

        # 5️⃣ Generate predictions
        if self.model_type == "logistic":
            preds_df = generate_today_predictions(threshold=self.threshold)
        else:
            df_new = build_features_for_new_games("data/new_games.csv")
            feature_cols = ["AGE","PTS","AST","REB","GAMES_PLAYED","PTS_per_AST","REB_rate"]
            data = df_new[feature_cols].fillna(0).to_numpy()
            games = list(zip(df_new["TEAM_HOME"], df_new["TEAM_AWAY"]))
            home_team_odds = df_new["american_odds"].tolist()
            away_team_odds = df_new["american_odds"].tolist()
            todays_games_uo = df_new.get("OU", [200] * len(df_new))

            if self.model_type == "nn":
                results = nn_runner(data, todays_games_uo, df_new[feature_cols], games,
                                    home_team_odds, away_team_odds, kelly_criterion=True)
            elif self.model_type == "xgb":
                results = xgb_runner(data, todays_games_uo, df_new[feature_cols], games,
                                     home_team_odds, away_team_odds, kelly_criterion=True)
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            preds_df = pd.DataFrame(results)

        if preds_df.empty:
            logger.warning("No predictions available today.")
            return None, None
        logger.info("✅ Predictions generated")

        # 6️⃣ Generate picks
        generate_picks(preds_file=PREDICTIONS_FILE, out_file=PICKS_FILE)
        logger.info("✅ Picks generated")

        # 7️⃣ Simulate bankroll (adds EV + Kelly_Bet columns)
        history, metrics = simulate_bankroll(
            preds_df,
            strategy=self.strategy,
            max_fraction=self.max_fraction,
            bankroll=1000.0   # starting bankroll
        )

        # 8️⃣ Compute win rate
        total_games = len(preds_df)
        correct = sum(preds_df["winner"] == preds_df["home_team"]) if "winner" in preds_df else 0
        win_rate = round((correct / total_games) * 100, 2) if total_games > 0 else 0

        # 9️⃣ Save predictions + summary
        preds_df.to_csv(PICKS_BANKROLL_FILE, index=False)

        summary = pd.DataFrame([{
            "Summary": "Daily Results",
            "Final_Bankroll": preds_df["bankroll"].iloc[-1],
            "Win_Rate": f"{win_rate}%",
            "Avg_EV": round(preds_df["EV"].mean(), 2),
            "Avg_Kelly_Bet": round(preds_df["Kelly_Bet"].mean(), 2)
        }])

        summary.to_csv(PICKS_BANKROLL_FILE, mode="a", header=True, index=False)
        logger.info(f"✅ Daily summary saved to {PICKS_BANKROLL_FILE}")

        return preds_df, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the NBA prediction pipeline")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--strategy", type=str, default="kelly")
    parser.add_argument("--max_fraction", type=float, default=0.05)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--sportsbook", type=str, default="fanduel")
    parser.add_argument("--model_type", type=str, default="logistic",
                        choices=["logistic", "nn", "xgb"])
    args = parser.parse_args()

    pipeline = PredictionPipeline(
        threshold=args.threshold,
        strategy=args.strategy,
        max_fraction=args.max_fraction,
        results_dir=args.results_dir,
        use_synthetic=args.use_synthetic,
        sportsbook=args.sportsbook,
        model_type=args.model_type
    )

    preds_df, metrics = pipeline.run()
    if preds_df is not None:
        logger.info("✅ Pipeline finished successfully")
        logger.info(f"Metrics: {metrics}")