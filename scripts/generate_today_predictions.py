# scripts/generate_today_predictions.py
import pandas as pd
from scripts.utils import setup_logger

logger = setup_logger("generate_today_predictions")

def generate_today_predictions(threshold=0.6, cli=False, notify=False, outdir="results"):
    """
    Generate predictions for today's games.
    Currently returns dummy predictions.
    """
    import os
    os.makedirs(outdir, exist_ok=True)

    logger.info("Generating today's predictions...")
    df = pd.DataFrame({
        "game_id": [101, 102, 103],
        "home_team": ["Team X", "Team Y", "Team Z"],
        "away_team": ["Team A", "Team B", "Team C"],
        "pred_home_win_prob": [0.65, 0.55, 0.70],
        "decimal_odds": [1.8, 2.0, 1.9],
        "ev": [0.07, 0.02, 0.09]
    })

    # Filter strong picks
    df = df[df["pred_home_win_prob"] >= threshold]
    df.to_csv(f"{outdir}/predictions.csv", index=False)
    logger.info(f"Predictions saved to {outdir}/predictions.csv")
    return df
