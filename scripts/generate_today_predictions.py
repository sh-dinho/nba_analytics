# ============================================================
# File: scripts/generate_today_predictions.py
# Purpose: Generate today's predictions from trained model
# ============================================================

import os
import pandas as pd
import joblib
from core.config import MODEL_FILE_PKL, PREDICTIONS_FILE
from core.log_config import setup_logger
from core.utils import ensure_columns

logger = setup_logger("generate_today_predictions")


def generate_today_predictions(features_file: str, threshold: float = 0.6) -> pd.DataFrame:
    """
    Generate predictions for today's games using the trained model.
    Handles both numeric and categorical features.
    Saves predictions to PREDICTIONS_FILE and returns DataFrame.
    """
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"{features_file} not found. Run build_features_for_new_games first.")

    df = pd.read_csv(features_file)

    # Load trained model artifact
    if not os.path.exists(MODEL_FILE_PKL):
        raise FileNotFoundError(f"{MODEL_FILE_PKL} not found. Run train_model first.")

    artifact = joblib.load(MODEL_FILE_PKL)
    if isinstance(artifact, dict) and "model" in artifact:
        pipeline = artifact["model"]
        feature_cols = artifact["features"]
    else:
        pipeline = artifact
        feature_cols = [
            "home_avg_pts", "home_avg_ast", "home_avg_reb", "home_avg_games_played",
            "away_avg_pts", "away_avg_ast", "away_avg_reb", "away_avg_games_played",
            "home_team", "away_team"
        ]

    logger.info(f"✅ Loaded model from {MODEL_FILE_PKL}")

    # ✅ Make decimal_odds optional
    required = feature_cols + ["game_id", "home_team", "away_team"]
    if "decimal_odds" in df.columns:
        required.append("decimal_odds")
    else:
        logger.warning("⚠️ 'decimal_odds' column missing — skipping EV calculations.")

    ensure_columns(df, required, "game features")

    # Predict probabilities
    X = df[feature_cols]
    probs = pipeline.predict_proba(X)[:, 1]  # probability home team wins
    preds = (probs >= threshold).astype(int)

    # Build prediction DataFrame
    df["pred_home_win_prob"] = probs
    df["predicted_home_win"] = preds

    # Save predictions
    df.to_csv(PREDICTIONS_FILE, index=False)
    logger.info(f"📊 Predictions saved to {PREDICTIONS_FILE} ({len(df)} rows)")

    # Log game-level predictions
    logger.info("=== GAME-LEVEL PREDICTIONS ===")
    for _, row in df.iterrows():
        logger.info(
            f"{row['home_team']} vs {row['away_team']} → "
            f"Home win prob: {row['pred_home_win_prob']:.2f} | "
            f"Predicted: {'Home Win' if row['predicted_home_win'] else 'Away Win'}"
        )

    return df


if __name__ == "__main__":
    generate_today_predictions(PREDICTIONS_FILE)