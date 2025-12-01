# File: scripts/build_weekly_summary.py

import pandas as pd
import os
from nba_analytics_core.notifications import send_telegram_message

RESULTS_DIR = "results"
TRENDS_FILE = os.path.join(RESULTS_DIR, "player_trends.csv")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "weekly_summary.csv")
TEAM_SUMMARY_FILE = os.path.join(RESULTS_DIR, "team_summary.csv")

def build_weekly_summary(notify=False):
    if not os.path.exists(TRENDS_FILE):
        print("Trend data not found. Run player_trends.py first.")
        return

    df = pd.read_csv(TRENDS_FILE)

    # Player of the Week: biggest positive change across combined metrics
    df["total_change"] = df[["PTS_change","REB_change","AST_change","TS_PCT_change"]].sum(axis=1)
    player_of_week = df.loc[df["total_change"].idxmax()]
    top_faller = df.loc[df["total_change"].idxmin()]

    # Team of the Week: aggregate improvements by team
    team_changes = df.groupby("TEAM_ABBREVIATION")[["PTS_change","REB_change","AST_change","TS_PCT_change"]].mean()
    team_changes["total_change"] = team_changes.sum(axis=1)
    team_of_week = team_changes.loc[team_changes["total_change"].idxmax()]

    # Save team summary
    os.makedirs(RESULTS_DIR, exist_ok=True)
    team_changes.to_csv(TEAM_SUMMARY_FILE)

    # Build weekly summary
    summary = pd.DataFrame([{
        "player_of_week": player_of_week["PLAYER_NAME"],
        "team_of_week": team_of_week.name,
        "player_pts_change": player_of_week["PTS_change"],
        "player_reb_change": player_of_week["REB_change"],
        "player_ast_change": player_of_week["AST_change"],
        "player_eff_change": player_of_week["TS_PCT_change"],
        "top_faller": top_faller["PLAYER_NAME"],
        "team_pts_change": team_of_week["PTS_change"],
        "team_reb_change": team_of_week["REB_change"],
        "team_ast_change": team_of_week["AST_change"],
        "team_eff_change": team_of_week["TS_PCT_change"]
    }])

    summary.to_csv(SUMMARY_FILE, index=False)
    print(f"✅ Weekly summary saved to {SUMMARY_FILE}")
    print(f"📂 Team summary saved to {TEAM_SUMMARY_FILE}")

    # Telegram notification
    if notify:
        msg = (
            f"📊 Weekly Summary\n"
            f"Player of the Week: {player_of_week['PLAYER_NAME']} (+{player_of_week['total_change']:.1f} impact)\n"
            f"Top Faller: {top_faller['PLAYER_NAME']} ({top_faller['total_change']:.1f} impact)\n"
            f"Team of the Week: {team_of_week.name} (+{team_of_week['total_change']:.1f} impact)"
        )
        send_telegram_message(msg)

if __name__ == "__main__":
    build_weekly_summary(notify=True)