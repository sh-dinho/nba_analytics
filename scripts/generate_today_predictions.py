# ============================================================
# File: scripts/generate_today_predictions.py
# Purpose: Generate today's predictions from trained model
# ============================================================

import os
import pandas as pd
import joblib
from core.config import MODEL_FILE_PKL, PREDICTIONS_FILE, DEFAULT_BANKROLL, MAX_KELLY_FRACTION, PICKS_FILE
from core.log_config import setup_logger
from core.utils import ensure_columns

logger = setup_logger("generate_today_predictions")


def generate_today_predictions(features_file: str, threshold: float = 0.6) -> pd.DataFrame:
    """
    Generate predictions for today's games using the trained model.
    Handles binary, regression, and multi-class targets.
    Saves predictions to PREDICTIONS_FILE and returns DataFrame.
    """
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"{features_file} not found. Run build_features_for_new_games first.")

    df = pd.read_csv(features_file)

    # Load trained model artifact
    if not os.path.exists(MODEL_FILE_PKL):
        raise FileNotFoundError(f"{MODEL_FILE_PKL} not found. Run train_model first.")

    artifact = joblib.load(MODEL_FILE_PKL)
    pipeline = artifact["model"]
    feature_cols = artifact["features"]
    target = artifact.get("target", "label")

    logger.info(f"✅ Loaded model (target={target}) from {MODEL_FILE_PKL}")

    # Required columns
    required = feature_cols + ["game_id", "home_team", "away_team"]
    if "decimal_odds" in df.columns:
        required.append("decimal_odds")
    ensure_columns(df, required, "game features")

    X = df[feature_cols]

    # Predict depending on target type
    if target == "label":
        probs = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, "predict_proba") else pipeline.predict(X)
        preds = (probs >= threshold).astype(int)
        df["pred_home_win_prob"] = probs
        df["predicted_home_win"] = preds

    elif target == "margin":
        preds = pipeline.predict(X)
        df["predicted_margin"] = preds
        # Convert margin prediction into implied win probability (sigmoid approximation)
        df["pred_home_win_prob"] = 1 / (1 + pd.np.exp(-0.1 * preds))
        df["predicted_home_win"] = (df["pred_home_win_prob"] >= threshold).astype(int)

    elif target == "outcome_category":
        preds = pipeline.predict(X)
        df["predicted_outcome_category"] = preds
        # If predict_proba available, log probabilities
        if hasattr(pipeline, "predict_proba"):
            prob_df = pd.DataFrame(pipeline.predict_proba(X), columns=pipeline.classes_)
            df = pd.concat([df, prob_df], axis=1)

    # Save predictions
    df.to_csv(PREDICTIONS_FILE, index=False)
    logger.info(f"📊 Predictions saved to {PREDICTIONS_FILE} ({len(df)} rows)")

    # Picks logic only applies to binary/derived win probability
    picks = []
    if "decimal_odds" in df.columns and "pred_home_win_prob" in df.columns:
        logger.info("=== GAME-LEVEL PREDICTIONS WITH PICKS & STAKING ===")
        for _, row in df.iterrows():
            odds = row["decimal_odds"]
            p = row["pred_home_win_prob"]
            q = 1 - p
            b = odds - 1
            ev = p * odds - 1
            kelly_fraction = (b * p - q) / b if b > 0 else 0
            if kelly_fraction > 0:
                kelly_fraction = min(kelly_fraction, MAX_KELLY_FRACTION)
                stake_amount = DEFAULT_BANKROLL * kelly_fraction
                picks.append({
                    "game_id": row["game_id"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "pred_home_win_prob": row.get("pred_home_win_prob"),
                    "predicted_home_win": row.get("predicted_home_win"),
                    "decimal_odds": odds,
                    "expected_value": ev,
                    "kelly_fraction": kelly_fraction,
                    "stake_amount": stake_amount,
                })
                logger.info(f"{row['home_team']} vs {row['away_team']} → EV={ev:.3f} | ✅ Pick | Stake={stake_amount:.2f}")

    if picks:
        pd.DataFrame(picks).to_csv(PICKS_FILE, index=False)
        logger.info(f"💾 Picks saved to {PICKS_FILE} ({len(picks)} rows)")
    else:
        logger.info("ℹ️ No positive EV picks found today.")

    return df


if __name__ == "__main__":
    generate_today_predictions(PREDICTIONS_FILE)