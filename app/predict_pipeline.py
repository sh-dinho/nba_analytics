# app/predict_pipeline.py
import pandas as pd
import joblib
import os
from scripts.utils import setup_logger

logger = setup_logger("predict_pipeline")

def generate_today_predictions(
    features_file: str = "data/training_features.csv",
    model_file: str = "models/game_predictor.pkl",
    threshold: float = 0.6,
    cli: bool = False,
    notify: bool = False,
    outdir: str = "results"
) -> pd.DataFrame:
    """
    Generate predictions for today's games using a trained ML model.

    Args:
        features_file: CSV file with today's game features.
        model_file: Path to the trained model.
        threshold: Minimum probability to consider a strong pick.
        cli: If True, run in CLI mode (minimal logging).
        notify: Whether to send Telegram notifications (optional).
        outdir: Where to save predictions CSV.

    Returns:
        pd.DataFrame: Predictions with columns:
            - game_id
            - home_team
            - away_team
            - pred_home_win_prob
            - decimal_odds
            - ev
    """
    os.makedirs(outdir, exist_ok=True)

    # ----------------------
    # Load features
    # ----------------------
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found: {features_file}")
    features = pd.read_csv(features_file)
    if features.empty:
        raise ValueError(f"No features found in {features_file}")
    logger.info(f"Loaded features for {len(features)} games from {features_file}")

    # ----------------------
    # Load trained model
    # ----------------------
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Trained model not found: {model_file}")
    model = joblib.load(model_file)
    logger.info(f"Loaded trained model from {model_file}")

    # ----------------------
    # Make predictions
    # ----------------------
    # Assuming model.predict_proba gives [prob_lose, prob_win]
    probs = model.predict_proba(features)[:,1]
    features["pred_home_win_prob"] = probs

    # ----------------------
    # EV calculation
    # ----------------------
    if "decimal_odds" not in features.columns:
        features["decimal_odds"] = 2.0  # default dummy odds

    features["ev"] = features["pred_home_win_prob"] * (features["decimal_odds"] - 1) - (1 - features["pred_home_win_prob"])

    # ----------------------
    # Filter strong picks
    # ----------------------
    strong_preds = features[features["pred_home_win_prob"] >= threshold].copy()
    strong_preds = strong_preds.reset_index(drop=True)
    logger.info(f"{len(strong_preds)} strong picks above threshold {threshold}")

    # ----------------------
    # Save predictions
    # ----------------------
    out_file = os.path.join(outdir, "predictions.csv")
    strong_preds.to_csv(out_file, index=False)
    logger.info(f"Predictions saved to {out_file}")

    return strong_preds

# ----------------------
# CLI entry
# ----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate today's predictions from trained model")
    parser.add_argument("--features", type=str, default="data/training_features.csv")
    parser.add_argument("--model", type=str, default="models/game_predictor.pkl")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    generate_today_predictions(
        features_file=args.features,
        model_file=args.model,
        threshold=args.threshold,
        outdir=args.outdir,
        cli=True
    )
