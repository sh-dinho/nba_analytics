# ============================================================
# File: src/prediction_engine/predictor.py
# Purpose: Predict win probabilities using trained model with logging + CLI
# Project: nba_analysis
# Version: 1.3 (fixes CLI flag, numeric coercion, feature alignment, alias)
# ============================================================

import argparse
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import logging
import mlflow

from src.utils.logging_config import configure_logging


class NBAPredictor:
    def __init__(self, model_path: str, log_level: str = "INFO", log_dir: str = "logs"):
        self.model_path = Path(model_path)
        self.logger = configure_logging(
            level=log_level, log_dir=log_dir, name="predictor"
        )

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

        # For sklearn models/pipelines, feature_names_in_ may exist
        self.expected_features = getattr(self.model, "feature_names_in_", None)

    def _validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("Features must be a pandas DataFrame.")
        if features.empty:
            raise ValueError("Features DataFrame is empty.")

        # Drop known non-feature identifier columns; extend as needed
        drop_cols = [
            "win",
            "unique_id",
            "prediction_date",
            "TEAM_NAME",
            "OPPONENT_TEAM_NAME",
            "GAME_ID",
        ]
        features = features.drop(
            columns=[c for c in drop_cols if c in features.columns], errors="ignore"
        )

        # Coerce everything to numeric to avoid dtype issues
        features = features.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Align with model's expected features if available
        if self.expected_features is not None:
            missing = [f for f in self.expected_features if f not in features.columns]
            if missing:
                self.logger.warning(f"Missing expected features: {missing}")
            features = features.reindex(
                columns=list(self.expected_features), fill_value=0
            )

        return features

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        features = self._validate_features(features)
        self.logger.info(
            f"Generating probability predictions for {len(features)} samples"
        )

        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Loaded model does not support predict_proba.")

        try:
            proba = self.model.predict_proba(features)[:, 1]
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

        proba_series = pd.Series(proba, index=features.index, name="win_proba")
        self.logger.info(f"Average win probability: {proba_series.mean():.3f}")
        self.logger.info(
            f"Min: {proba_series.min():.3f}, Max: {proba_series.max():.3f}, Std: {proba_series.std():.3f}"
        )
        return proba_series

    def predict_label(
        self, features: pd.DataFrame, threshold: float = 0.5
    ) -> pd.Series:
        features = self._validate_features(features)
        self.logger.info(f"Generating label predictions with threshold={threshold}")

        proba = self.predict_proba(features)
        labels = (proba >= threshold).astype(int).rename("win_pred")
        win_rate = float(labels.mean())
        self.logger.info(f"Predicted win rate: {win_rate:.3f}")
        return labels


# Backward-compatible alias to match existing imports in your codebase
class Predictor(NBAPredictor):
    pass


# -------------------------------
# CLI Wrapper
# -------------------------------


def main():
    parser = argparse.ArgumentParser(description="NBA Predictor CLI")
    parser.add_argument(
        "--model", required=True, help="Path to trained model file (.pkl)"
    )
    parser.add_argument(
        "--features", required=True, help="Path to CSV or Parquet file with features"
    )
    parser.add_argument("--mode", choices=["proba", "label"], default="proba")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", help="Optional path to save predictions as CSV")
    parser.add_argument(
        "--mlflow", action="store_true", help="Log predictions to MLflow"
    )  # fixed flag
    args = parser.parse_args()

    try:
        predictor = NBAPredictor(model_path=args.model)

        feat_path = Path(args.features)
        if feat_path.suffix.lower() == ".parquet":
            features = pd.read_parquet(feat_path)
        else:
            features = pd.read_csv(feat_path)

        if args.mode == "proba":
            preds = predictor.predict_proba(features)
            summary = f"Summary: mean={preds.mean():.3f}, min={preds.min():.3f}, max={preds.max():.3f}"
        else:
            preds = predictor.predict_label(features, threshold=args.threshold)
            summary = f"Summary: win_rate={preds.mean():.3f}"

        print(preds.head())
        print(f"... total {len(preds)} predictions")
        print(summary)

        if args.output:
            out_path = Path(args.output).resolve()
            preds.to_csv(out_path, index=True)
            print(f"Predictions saved to {out_path}")

        if args.mlflow:
            model_abs = Path(args.model).resolve()
            feats_abs = Path(args.features).resolve()
            with mlflow.start_run(run_name="predictor_cli", nested=True):
                mlflow.log_param("model_path", str(model_abs))
                mlflow.log_param("features_file", str(feats_abs))
                mlflow.log_metric("mean_prediction", float(preds.mean()))
                mlflow.log_artifact(str(model_abs), artifact_path="model")
                if args.output:
                    mlflow.log_artifact(
                        str(Path(args.output).resolve()), artifact_path="predictions"
                    )

    except Exception as e:
        print(f"❌ Prediction failed: {e}")


if __name__ == "__main__":
    main()
