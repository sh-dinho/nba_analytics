# File: scripts/generate_picks.py

import pandas as pd
import os
import logging
import datetime

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

PICKS_LOG = os.path.join(RESULTS_DIR, "picks_summary.csv")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def main(preds_file=f"{RESULTS_DIR}/predictions.csv", out_file=f"{RESULTS_DIR}/picks.csv"):
    """
    Generate picks from predictions using a simple EV strategy.
    Adds a rolling summary log for tracking.
    """
    if not os.path.exists(preds_file):
        raise FileNotFoundError(f"{preds_file} not found.")

    df = pd.read_csv(preds_file)

    # Simple strategy: pick HOME if prob > 0.5, else AWAY
    df["pick"] = df.apply(lambda row: "HOME" if row.pred_home_win_prob > 0.5 else "AWAY", axis=1)

    # Save picks
    df.to_csv(out_file, index=False)
    logging.info(f"✅ Picks saved to {out_file} | Total picks: {len(df)}")

    # Summary stats
    home_picks = (df["pick"] == "HOME").sum()
    away_picks = (df["pick"] == "AWAY").sum()
    avg_ev = df["ev"].mean() if "ev" in df.columns else None

    logging.info(
        f"📊 Picks summary: HOME={home_picks}, AWAY={away_picks}, Avg EV={avg_ev:.3f if avg_ev else 0}"
    )

    # Append to rolling log
    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_entry = pd.DataFrame([{
        "timestamp": run_time,
        "total_picks": len(df),
        "home_picks": home_picks,
        "away_picks": away_picks,
        "avg_ev": avg_ev
    }])

    if os.path.exists(PICKS_LOG):
        summary_entry.to_csv(PICKS_LOG, mode="a", header=False, index=False)
    else:
        summary_entry.to_csv(PICKS_LOG, index=False)

    logging.info(f"📁 Picks summary appended to {PICKS_LOG}")

    return df

if __name__ == "__main__":
    main()