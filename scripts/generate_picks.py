# scripts/generate_picks.py
import pandas as pd
from scripts.utils import setup_logger
import os

logger = setup_logger("generate_picks")

def main(preds_file="results/predictions.csv", odds_file=None, out_file="results/picks.csv", notify=False):
    """
    Generate picks from predictions.
    For now, this just copies predictions to picks file.
    """
    if not os.path.exists(preds_file):
        raise FileNotFoundError(f"Picks file not found: {preds_file}")

    df = pd.read_csv(preds_file)

    # Add a 'pick' column: pick home if prob > 0.5
    df["pick"] = df["pred_home_win_prob"].apply(lambda x: "home" if x > 0.5 else "away")

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    df.to_csv(out_file, index=False)
    logger.info(f"Picks saved to {out_file}")

    return df
