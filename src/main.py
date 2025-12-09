# ============================================================
# File: src/main.py
# Purpose: CLI entrypoint for NBA analysis pipeline
# ============================================================

import argparse
from pathlib import Path

import pandas as pd

from src.model_training.training import train_logreg
from features.game_features import generate_features_for_games
from src.prediction_engine.predictor import NBAPredictor
from src.utils.io import load_dataframe, save_dataframe
from src.utils.logging import configure_logging

logger = configure_logging(level="INFO", log_dir="logs", name="main")


def run_pipeline(game_ids: list[str], season: int, out_dir: str = "results"):
    """Run full pipeline: feature generation → training → prediction."""
    logger.info("Starting NBA analysis pipeline...")

    # --- Step 1: Generate features ---
    features = generate_features_for_games(game_ids, season=season)
    features_path = Path(out_dir) / "features.parquet"
    save_dataframe(features, features_path)
    logger.info(f"Features saved to {features_path}")

    # --- Step 2: Train model ---
    result = train_logreg(str(features_path), out_dir="models")
    model_path = result["model_path"]
    logger.info(f"Model trained and saved to {model_path}")

    # --- Step 3: Predict outcomes ---
    df_loaded = load_dataframe(features_path)
    X = df_loaded.drop(columns=["game_id"], errors="ignore")
    predictor = NBAPredictor(model_path)
    proba = predictor.predict_proba(X)
    labels = predictor.predict_label(X)

    predictions = pd.DataFrame(
        {
            "game_id": df_loaded.get("game_id", range(len(X))),
            "win_proba": proba,
            "win_pred": labels,
        }
    )

    pred_path = Path(out_dir) / "predictions.csv"
    predictions.to_csv(pred_path, index=False)
    logger.info(f"Predictions saved to {pred_path}")

    logger.info("Pipeline complete.")
    return pred_path


def main():
    parser = argparse.ArgumentParser(description="NBA Analysis Pipeline")
    parser.add_argument(
        "--game_ids",
        type=str,
        default="0042300401,0022300649",
        help="Comma-separated list of NBA game IDs",
    )
    parser.add_argument(
        "--season", type=int, default=2023, help="NBA season year (e.g., 2023)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Output directory for features and predictions",
    )
    args = parser.parse_args()

    game_ids = args.game_ids.split(",")
    run_pipeline(game_ids, args.season, args.out_dir)


if __name__ == "__main__":
    main()
