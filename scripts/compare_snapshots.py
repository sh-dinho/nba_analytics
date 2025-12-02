# File: scripts/compare_snapshots.py (enhanced with weekly summary update)
import pandas as pd
import os
import json
from datetime import datetime
from nba_analytics_core.notifications import send_telegram_message

RESULTS_DIR = "results"
CURRENT_FILE = os.path.join(RESULTS_DIR, "player_leaderboards_current.csv")
PREVIOUS_FILE = os.path.join(RESULTS_DIR, "player_leaderboards_previous.csv")
TRENDS_FILE = os.path.join(RESULTS_DIR, "player_trends.csv")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "player_trends_summary.csv")
WEEKLY_SUMMARY_FILE = os.path.join(RESULTS_DIR, "weekly_summary.csv")
METADATA_FILE = os.path.join(RESULTS_DIR, "player_trends_meta.json")

REQUIRED_COLUMNS = {"PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "REB", "AST", "TS_PCT"}

def _ensure_columns(df, required_cols, name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

def build_weekly_summary(trends_df, notify=False):
    """
    Aggregate player trends into weekly team-level summary.
    """
    df = trends_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["week"] = df["date"].dt.isocalendar().week

    # Aggregation per team per week
    summary = df.groupby(["TEAM_ABBREVIATION", "week"]).agg({
        "PTS": "mean",
        "REB": "mean",
        "AST": "mean",
        "TS_PCT": "mean"
    }).reset_index()

    # Difference vs previous week
    summary = summary.sort_values(["TEAM_ABBREVIATION", "week"])
    for col in ["PTS", "REB", "AST", "TS_PCT"]:
        summary[f"{col}_diff_prev_week"] = summary.groupby("TEAM_ABBREVIATION")[col].diff()

    summary.to_csv(WEEKLY_SUMMARY_FILE, index=False)
    if notify:
        print(f"✅ Weekly summary updated at {WEEKLY_SUMMARY_FILE}")
    return summary

def compare_snapshots(notify=False, trend_threshold=2.0, update_weekly=True):
    if not os.path.exists(CURRENT_FILE) or not os.path.exists(PREVIOUS_FILE):
        print("❌ Missing snapshot files. Run weekly update first.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("📂 Loading snapshot files...")
    current = pd.read_csv(CURRENT_FILE)
    previous = pd.read_csv(PREVIOUS_FILE)

    # Validate required columns
    _ensure_columns(current, REQUIRED_COLUMNS, "player_leaderboards_current.csv")
    _ensure_columns(previous, REQUIRED_COLUMNS, "player_leaderboards_previous.csv")

    # Merge snapshots
    merged = current.merge(previous, on="PLAYER_NAME", suffixes=("_curr", "_prev"))

    metrics = ["PTS", "REB", "AST", "TS_PCT"]
    for metric in metrics:
        merged[f"{metric}_change"] = merged[f"{metric}_curr"] - merged[f"{metric}_prev"]

    # Composite impact change
    merged["impact_change"] = merged[[f"{m}_change" for m in ["PTS", "REB", "AST"]]].sum(axis=1)

    # Trend classification
    def classify_trend(x, threshold=trend_threshold):
        if x > threshold:
            return "Rising"
        elif x < -threshold:
            return "Falling"
        else:
            return "Stable"
    merged["trend"] = merged["impact_change"].apply(classify_trend)

    # Rank changes (PTS)
    merged["rank_curr"] = merged["PTS_curr"].rank(ascending=False)
    merged["rank_prev"] = merged["PTS_prev"].rank(ascending=False)
    merged["rank_change"] = merged["rank_prev"] - merged["rank_curr"]

    # Save detailed trends
    merged.to_csv(TRENDS_FILE, index=False)

    # Summary: top 10 risers
    summary = merged.sort_values("impact_change", ascending=False).head(10)
    summary.to_csv(SUMMARY_FILE, index=False)

    # Metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "rows": len(merged),
        "columns": merged.columns.tolist(),
        "current_file": CURRENT_FILE,
        "previous_file": PREVIOUS_FILE,
        "summary_file": SUMMARY_FILE
    }
    with open(METADATA_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Trends saved to {TRENDS_FILE}")
    print(f"✅ Summary exported to {SUMMARY_FILE}")
    print(f"📊 Rows: {len(merged)}, Columns: {len(merged.columns)}")
    print(f"🔎 Top Players: {list(summary['PLAYER_NAME'])}")

    # Telegram notification
    if notify and not summary.empty:
        top_riser = summary.iloc[0]
        top_faller = merged.sort_values("impact_change").iloc[0]
        msg = (
            f"📊 Weekly Player Trends\n"
            f"Top Riser: {top_riser['PLAYER_NAME']} (+{top_riser['impact_change']:.1f} impact)\n"
            f"Top Faller: {top_faller['PLAYER_NAME']} ({top_faller['impact_change']:.1f} impact)\n"
            f"Summary saved to {SUMMARY_FILE}"
        )
        try:
            send_telegram_message(msg)
            print("✅ Telegram notification sent")
        except Exception as e:
            print(f"⚠️ Failed to send Telegram notification: {e}")

    # Optional: update weekly summary automatically
    weekly_summary = None
    if update_weekly:
        weekly_summary = build_weekly_summary(merged, notify=notify)

    return merged, summary, weekly_summary

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare current vs previous player snapshots")
    parser.add_argument("--no-notify", action="store_true", help="Disable Telegram notifications")
    parser.add_argument("--threshold", type=float, default=2.0, help="Impact change threshold for trend classification")
    parser.add_argument("--no-weekly", action="store_true", help="Disable automatic weekly summary update")
    args = parser.parse_args()

    compare_snapshots(
        notify=not args.no_notify,
        trend_threshold=args.threshold,
        update_weekly=not args.no_weekly
    )
