"""
Predict today's NBA games using advanced pre‑game features.
"""

from loguru import logger
import pandas as pd
from pathlib import Path

from src.features.feature_builder import PreGameFeatureBuilder
from src.model.train import load_model, load_metadata


def predict_today():
    logger.info("Predicting today's NBA games...")

    today_path = Path("data/raw/today_games.csv")
    if not today_path.exists():
        raise FileNotFoundError("Run fetch_today_games.py first.")

    today_games = pd.read_csv(today_path)

    # Build pre‑game features
    builder = PreGameFeatureBuilder()
    df_features = builder.build_for_games(today_games)

    # Load model + metadata
    model = load_model()
    metadata = load_metadata()

    feature_cols = metadata["features_used"]
    df_features["probability"] = model.predict_proba(df_features[feature_cols])[:, 1]

    # Save predictions
    out = Path("data/predictions/today_predictions.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(out, index=False)

    logger.info(f"Saved today's predictions → {out}")
    return df_features


if __name__ == "__main__":
    predict_today()