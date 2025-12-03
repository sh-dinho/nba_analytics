# ============================================================
# File: scripts/generate_today_predictions.py
# Purpose: Generate predictions for today's games
# ============================================================

import os
import pandas as pd
import joblib
import datetime
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from core.config import MODEL_FILE_PKL, PREDICTIONS_FILE, NEW_GAMES_FILE
from core.log_config import setup_logger
from core.exceptions import PipelineError, DataError
from core.utils import ensure_columns
from scripts.build_features import build_features_for_new_games

logger = setup_logger("generate_today_predictions")


def generate_today_predictions(features_file: str | None = None, threshold: float = 0.6) -> pd.DataFrame:
    """Generate predictions for today's games and log headline metrics."""
    if not os.path.exists(MODEL_FILE_PKL):
        logger.error("❌ No trained model found. Train a model first.")
        raise FileNotFoundError(MODEL_FILE_PKL)

    try:
        saved = joblib.load(MODEL_FILE_PKL)
        model = saved["model"]
        feature_order = saved["features"]
        logger.info(f"✅ Loaded model and feature order from {MODEL_FILE_PKL}")
    except Exception as e:
        raise PipelineError(f"Failed to load model from {MODEL_FILE_PKL}: {e}")

    # Default to NEW_GAMES_FILE if no features_file provided
    if features_file is None:
        features_file = NEW_GAMES_FILE

    if not os.path.exists(features_file):
        raise FileNotFoundError(f"{features_file} not found. Run fetch_new_games.py first.")

    df = build_features_for_new_games(features_file)

    # Validate required features
    try:
        ensure_columns(df, set(feature_order), "new game features")
    except ValueError as e:
        raise DataError(str(e))

    X_num = df[feature_order]
    X_num = X_num.fillna(0).replace([float("inf"), -float("inf")], 0)
    logger.warning("⚠️ Using simple fillna(0) for missing/inf values. Implement trained Imputer/Scaler for production.")

    try:
        probs = model.predict_proba(X_num)[:, 1]
    except Exception as e:
        raise PipelineError(f"Prediction failed: {e}")

    df["pred_home_win_prob"] = probs
    df["predicted_home_win"] = (probs >= threshold).astype(int)
    df["Date"] = datetime.datetime.now().strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
    df.to_csv(PREDICTIONS_FILE, index=False)
    logger.info(f"📊 Predictions saved to {PREDICTIONS_FILE} ({len(df)} rows)")

    # --- Headline metrics ---
    if "label" in df.columns and not df["label"].isnull().all():
        y_true = df["label"].dropna()
        y_pred = df.loc[y_true.index, "predicted_home_win"]
        y_proba = df.loc[y_true.index, "pred_home_win_prob"]

        metrics = {}
        try:
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
        except Exception:
            metrics["accuracy"] = None
        try:
            metrics["brier"] = brier_score_loss(y_true, y_proba)
        except Exception:
            metrics["brier"] = None
        try:
            metrics["auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            metrics["auc"] = None

        logger.info("=== PREDICTION METRICS ===")
        for k, v in metrics.items():
            if v is not None:
                logger.info(f"{k.capitalize()}: {v:.3f}")
            else:
                logger.info(f"{k.capitalize()}: unavailable")
    else:
        logger.info("⚠️ No ground-truth labels available for metrics (new games only).")

    return df


if __name__ == "__main__":
    generate_today_predictions()