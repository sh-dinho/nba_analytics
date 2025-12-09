# ============================================================
# File: src/prediction_engine/predictor.py
# Purpose: Predict win probabilities using trained model with logging + CLI
# ============================================================

from pathlib import Path
import joblib
import pandas as pd
import argparse
from src.utils.logging import configure_logging


class NBAPredictor:
    def __init__(self, model_path: str, log_level="INFO", log_dir="logs"):
        self.model_path = Path(model_path)
        self.logger = configure_logging(level=log_level, log_dir=log_dir, name="predictor")

        if not self.model_path.exists():
            self.logger.error(f"Model not found at {self.model_path}")
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        self.logger.info(f"Loading model from {self.model_path}")
        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
        self.logger.info("Model successfully loaded.")

    def _validate_features(self, features: pd.DataFrame):
        if not isinstance(features, pd.DataFrame):
            raise TypeError("Features must be a pandas DataFrame.")
        if features.empty:
            raise ValueError("Features DataFrame is empty.")

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """Predict win probabilities for given features."""
        self._validate_features(features)
        self.logger.info(f"Generating probability predictions for {len(features)} samples")

        try:
            proba = self.model.predict_proba(features)[:, 1]
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

        self.logger.info(f"Average win probability: {proba.mean():.3f}")
        self.logger.info(f"Min: {proba.min():.3f}, Max: {proba.max():.3f}, Std: {proba.std():.3f}")
        return pd.Series(proba, index=features.index, name="win_proba")

    def predict_label(self, features: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """Predict win/loss labels based on threshold."""
        self._validate_features(features)
        self.logger.info(f"Generating label predictions with threshold={threshold}")

        proba = self.predict_proba(features)
        labels = (proba >= threshold).astype(int).rename("win_pred")
        win_rate = labels.mean()
        self.logger.info(f"Predicted win rate: {win_rate:.3f}")
        return labels


# -------------------------------
# CLI Wrapper
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="NBA Predictor CLI")
    parser.add_argument("--model", required=True, help="Path to trained model file (.pkl)")
    parser.add_argument("--features", required=True, help="Path to CSV file with features")
    parser.add_argument("--mode", choices=["proba", "label"], default="proba")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", help="Optional path to save predictions as CSV")
    args = parser.parse_args()

    predictor = NBAPredictor(model_path=args.model)
    features = pd.read_csv(args.features)

    if args.mode == "proba":
        preds = predictor.predict_proba(features)
    else:
        preds = predictor.predict_label(features, threshold=args.threshold)

    print(preds)
    if args.output:
        preds.to_csv(args.output, index=True)
        print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
