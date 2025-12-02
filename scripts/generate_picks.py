import pandas as pd
import os

def main(preds_file="results/predictions.csv", out_file="results/picks.csv", notify=False):
    print(f"🎯 Generating picks from {preds_file}")
    df = pd.read_csv(preds_file)
    # Simple pick logic: choose games with EV > 0
    df["pick"] = df["ev"].apply(lambda x: "HOME" if x > 0 else "AWAY")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"✅ Picks saved to {out_file}")
