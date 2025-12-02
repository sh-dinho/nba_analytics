import pandas as pd
import os

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def main(preds_file=f"{RESULTS_DIR}/predictions.csv", out_file=f"{RESULTS_DIR}/picks.csv"):
    """
    Generate picks from predictions and simple EV strategy.
    """
    if not os.path.exists(preds_file):
        raise FileNotFoundError(f"{preds_file} not found.")

    df = pd.read_csv(preds_file)
    df["pick"] = df.apply(lambda row: "HOME" if row.pred_home_win_prob > 0.5 else "AWAY", axis=1)
    df.to_csv(out_file, index=False)
    print(f"Picks saved to {out_file}")
    return df
