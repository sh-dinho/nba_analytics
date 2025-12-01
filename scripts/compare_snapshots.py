# File: scripts/compare_snapshots.py
import pandas as pd
import os
from nba_analytics_core.notifications import send_telegram_message

RESULTS_DIR = "results"
CURRENT_FILE = os.path.join(RESULTS_DIR, "player_leaderboards_current.csv")
PREVIOUS_FILE = os.path.join(RESULTS_DIR, "player_leaderboards_previous.csv")
TRENDS_FILE = os.path.join(RESULTS_DIR, "player_trends.csv")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "player_trends_summary.csv")

def compare_snapshots(notify=False):
    if not os.path.exists(CURRENT_FILE) or not os.path.exists(PREVIOUS_FILE):
        print("Missing snapshot files. Run weekly update first.")
        return

    current = pd.read_csv(CURRENT_FILE)
    previous = pd.read_csv(PREVIOUS_FILE)

    # Merge snapshots
    merged = current.merge(previous, on="PLAYER_NAME", suffixes=("_curr", "_prev"))

    # Calculate changes across multiple metrics
    for metric in ["PTS", "REB", "AST", "TS_PCT"]:
        merged[f"{metric}_change"] = merged[f"{metric}_curr"] - merged[f"{metric}_prev"]

    # Composite impact change
    merged["impact_change"] = merged["PTS_change"] + merged["REB_change"] + merged["AST_change"]

    # Trend classification
    merged["trend"] = merged["impact_change"].apply(
        lambda x: "Rising" if x > 2 else ("Falling" if x < -2 else "Stable")
    )

    # Rank changes (scoring rank)
    current["scoring_rank"] = current["PTS"].rank(ascending=False)
    previous["scoring_rank"] = previous["PTS"].rank(ascending=False)
    merged = merged.merge(current[["PLAYER_NAME", "scoring_rank"]], on="PLAYER_NAME")
    merged = merged.merge(previous[["PLAYER_NAME", "scoring_rank"]], on="PLAYER_NAME", suffixes=("_curr", "_prev"))
    merged["rank_change"] = merged["scoring_rank_prev"] - merged["scoring_rank_curr"]

    # Save detailed + summary
    os.makedirs(RESULTS_DIR, exist_ok=True)
    merged.to_csv(TRENDS_FILE, index=False)
    summary = merged.sort_values("impact_change", ascending=False).head(10)
    summary.to_csv(SUMMARY_FILE, index=False)

    print(f"📊 Trends saved to {TRENDS_FILE} and summary exported to {SUMMARY_FILE}")

    # Telegram notification
    if notify and not summary.empty:
        top_riser = summary.iloc[0]
        top_faller = merged.sort_values("impact_change").iloc[0]
        msg = (
            f"📊 Weekly Player Trends\n"
            f"Top Riser: {top_riser['PLAYER_NAME']} (+{top_riser['impact_change']:.1f} impact)\n"
            f"Top Faller: {top_faller['PLAYER_NAME']} ({top_faller['impact_change']:.1f} impact)"
        )
        send_telegram_message(msg)

if __name__ == "__main__":
    compare_snapshots(notify=True)