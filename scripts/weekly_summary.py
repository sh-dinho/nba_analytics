import pandas as pd
import os

RESULTS_DIR = "results"
TRENDS_FILE = os.path.join(RESULTS_DIR, "player_trends.csv")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "weekly_summary.csv")

def build_weekly_summary():
    if not os.path.exists(TRENDS_FILE):
        print("Trend data not found. Run player_trends.py first.")
        return

    df = pd.read_csv(TRENDS_FILE)

    # Player of the Week: biggest positive change across combined metrics
    df["total_change"] = df[["PTS_change","REB_change","AST_change","TS_PCT_change"]].sum(axis=1)
    player_of_week = df.loc[df["total_change"].idxmax()]

    # Team of the Week: aggregate improvements by team
    team_changes = df.groupby("TEAM_ABBREVIATION")[["PTS_change","REB_change","AST_change","TS_PCT_change"]].mean()
    team_changes["total_change"] = team_changes.sum(axis=1)
    team_of_week = team_changes["total_change"].idxmax()

    summary = pd.DataFrame([{
        "player_of_week": player_of_week["PLAYER_NAME"],
        "team_of_week": team_of_week,
        "player_pts_change": player_of_week["PTS_change"],
        "player_reb_change": player_of_week["REB_change"],
        "player_ast_change": player_of_week["AST_change"],
        "player_eff_change": player_of_week["TS_PCT_change"]
    }])

    summary.to_csv(SUMMARY_FILE, index=False)
    print(f"✅ Weekly summary saved to {SUMMARY_FILE}")

if __name__ == "__main__":
    build_weekly_summary()