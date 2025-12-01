import pandas as pd
import os

RESULTS_DIR = "results"
CURRENT_FILE = os.path.join(RESULTS_DIR, "player_leaderboards_current.csv")
PREVIOUS_FILE = os.path.join(RESULTS_DIR, "player_leaderboards_previous.csv")
TRENDS_FILE = os.path.join(RESULTS_DIR, "player_trends.csv")

def compare_snapshots():
    if not os.path.exists(CURRENT_FILE) or not os.path.exists(PREVIOUS_FILE):
        print("Missing snapshot files. Run weekly update first.")
        return

    current = pd.read_csv(CURRENT_FILE)
    previous = pd.read_csv(PREVIOUS_FILE)

    merged = current.merge(previous, on="PLAYER_NAME", suffixes=("_curr", "_prev"))

    # Calculate changes across multiple metrics
    for metric in ["PTS", "REB", "AST", "TS_PCT"]:
        merged[f"{metric}_change"] = merged[f"{metric}_curr"] - merged[f"{metric}_prev"]

    # Trend classification based on points change
    merged["trend"] = merged["PTS_change"].apply(
        lambda x: "Rising" if x > 1 else ("Falling" if x < -1 else "Stable")
    )

    merged.to_csv(TRENDS_FILE, index=False)
    print(f"📊 Multi-metric trends saved to {TRENDS_FILE}")

if __name__ == "__main__":
    compare_snapshots()