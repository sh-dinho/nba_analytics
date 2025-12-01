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

    merged["PTS_change"] = merged["PTS_curr"] - merged["PTS_prev"]
    merged["REB_change"] = merged["REB_curr"] - merged["REB_prev"]
    merged["AST_change"] = merged["AST_curr"] - merged["AST_prev"]

    # Rank changes
    merged["trend"] = merged["PTS_change"].apply(
        lambda x: "Rising" if x > 1 else ("Falling" if x < -1 else "Stable")
    )

    merged.to_csv(TRENDS_FILE, index=False)
    print(f"📊 Trends saved to {TRENDS_FILE}")

if __name__ == "__main__":
    compare_snapshots()