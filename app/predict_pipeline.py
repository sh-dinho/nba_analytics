import pandas as pd
from scripts.utils import setup_logger, safe_mkdir
from scripts.build_features import main as build_features
from scripts.train_model import main as train_model
from scripts.generate_today_predictions import generate_today_predictions
from scripts.generate_picks import main as generate_picks
from scripts.simulate_bankroll import simulate_bankroll

logger = setup_logger("predict_pipeline")

class PredictionPipeline:
    def __init__(self, features_file="data/training_features.csv",
                 model_file="models/game_predictor.pkl",
                 predictions_dir="results",
                 threshold=0.6,
                 max_fraction=0.05,
                 strategy="kelly"):
        self.features_file = features_file
        self.model_file = model_file
        self.predictions_dir = predictions_dir
        self.threshold = threshold
        self.max_fraction = max_fraction
        self.strategy = strategy
        safe_mkdir(predictions_dir)

    def run(self, train_model_flag=True):
        # =======================
        # 1) Build Features
        # =======================
        df_feat = build_features(out_file=self.features_file)
        logger.info(f"Features ready: {self.features_file}")

        # =======================
        # 2) Train Model
        # =======================
        metrics = None
        if train_model_flag:
            metrics = train_model(features_file=self.features_file, model_file=self.model_file)
            logger.info(f"Model trained. Metrics: {metrics}")

        # =======================
        # 3) Generate Predictions
        # =======================
        df_preds = generate_today_predictions(
            model_file=self.model_file,
            features_file=self.features_file,
            threshold=self.threshold,
            outdir=self.predictions_dir
        )
        logger.info(f"Predictions generated: {len(df_preds)} rows")

        # =======================
        # 4) Generate Picks
        # =======================
        df_picks = generate_picks(preds_file=f"{self.predictions_dir}/predictions.csv",
                                  out_file=f"{self.predictions_dir}/picks.csv",
                                  threshold=self.threshold)
        logger.info(f"Picks generated: {len(df_picks)} rows")

        # =======================
        # 5) Bankroll Simulation
        # =======================
        trajectory, bankroll_metrics = simulate_bankroll(df_preds, strategy=self.strategy, max_fraction=self.max_fraction)
        df_preds["bankroll"] = trajectory
        logger.info(f"Bankroll simulation completed. Final bankroll: {bankroll_metrics['final_bankroll']}")

        return df_preds, df_picks, bankroll_metrics, metrics
