# File: scripts/compare_snapshots.py (production-ready)
import os
import json
import pandas as pd
from datetime import datetime
import logging
from nba_analytics_core.notifications import send_telegram_message

# ----------------------------
# Configurable paths
# ----------------------------
RESULTS_DIR = "results"
CURRENT_FILE = os.path.join(RESULTS_DIR, "player_leaderboards_current.csv")
PREVIOUS_FILE = os.path.join(RESULTS_DIR, "player_leaderboards_previous.csv")
TRENDS_FILE = os.path.join(RESULTS_DIR, "player_trends.csv")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "player_trends_summary.csv")
WEEKLY_SUMMARY_FILE = os.path.join(RESULTS_DIR, "weekly_summary.csv")
METADATA_FILE = os.path.join(RESULTS_DIR, "player_trends_meta.json")

REQUIRED_COLUMNS = {"PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "REB", "AST", "TS_PCT"}

# ----------------------------
# Logging setup
# ----------------------------
logger = logging.getLogger("compare_snapshots")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)

# ----------------------------
# Helper functions
# ----------------------------
def _ensure_columns(df: pd.DataFrame, required_cols: set, name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

def classify_trend(impact_change: float, threshold: float = 2.0) -> str:
    if impact_change > threshold:
        return "Rising"
    elif impact_change < -threshold:
        return "Falling"
    else:
        return "Stable"

def build_weekly_summary(trends_df: pd.DataFrame, notify: bool = False) -> pd.DataFrame:
    """
    Aggregate player trends into weekly team-level summary.
    """
    df = trends_df.copy()
    if "date" not in df.columns:
        logger.warning("No 'date' column found in trends dataframe. Weekly summary will be skipped.")
        return pd.DataFrame()  # Return empty

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        logger.warning("All 'date' values are invalid. Weekly summary skipped.")
        return pd.DataFrame()

    df["week"] = df["date"].dt.isocalendar().week

    summary = df.groupby(["TEAM_ABBREVIATION", "week"]).agg({
        "PTS": "mean",
        "REB": "mean",
        "AST": "mean",
        "TS_PCT": "mean"
    }).reset_index()

    # Week-over-week differences
    summary = summary.sort_values(["TEAM_ABBREVIATION", "week"])
    for col in ["PTS", "REB", "AST", "TS_PCT"]:
        summary[f"{col}_diff_prev_week"] = summary.groupby("TEAM_ABBREVIATION")[col].diff()

    summary.to_csv(WEEKLY_SUMMARY_FILE, index=False)
    if notify:
        logger.info(f"✅ Weekly summary saved to {WEEKLY_SUMMARY_FILE}")

    return summary

# ----------------------------
# Main function
# ----------------------------
def compare_snapshots(
    notify: bool = True,
    trend_threshold: float = 2.0,
    update_weekly: bool = True,
    fail_fast: bool = False
):
    """
    Compare current and previous player snapshots, classify trends, and generate summaries.

    Returns:
        merged_df, top_summary_df, weekly_summary_df
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(CURRENT_FILE) or not os.path.exists(PREVIOUS_FILE):
        msg = "Missing snapshot files. Run weekly update first."
        if fail_fast:
            raise FileNotFoundError(msg)
        logger.error(msg)
        return None, None, None

    try:
        logger.info("📂 Loading current and previous snapshot files...")
        current = pd.read_csv(CURRENT_FILE)
        previous = pd.read_csv(PREVIOUS_FILE)
        _ensure_columns(current, REQUIRED_COLUMNS, "player_leaderboards_current.csv")
        _ensure_columns(previous, REQUIRED_COLUMNS, "player_leaderboards_previous.csv")
    except Exception as e:
        if fail_fast:
            raise
        logger.error(f"Failed to load snapshots: {e}")
        return None, None, None

    # Merge snapshots
    merged = current.merge(previous, on="PLAYER_NAME", suffixes=("_curr", "_prev"))

    metrics = ["PTS", "REB", "AST", "TS_PCT"]
    for metric in metrics:
        merged[f"{metric}_change"] = merged[f"{metric}_curr"] - merged[f"{metric}_prev"]

    # Composite impact change
    merged["impact_change"] = merged[[f"{m}_change" for m in ["PTS", "REB", "AST"]]].sum(axis=1)

    # Trend classification
    merged["trend"] = merged["impact_change"].apply(lambda x: classify_trend(x, trend_threshold))

    # Rank changes
    merged["rank_curr"] = merged["PTS_curr"].rank(ascending=False)
    merged["rank_prev"] = merged["PTS_prev"].rank(ascending=False)
    merged["rank_change"] = merged["rank_prev"] - merged["rank_curr"]

    # Save full trends
    merged.to_csv(TRENDS_FILE, index=False)
    logger.info(f"✅ Trends saved to {TRENDS_FILE}")

    # Top 10 risers summary
    top_summary = merged.sort_values("impact_change", ascending=False).head(10)
    top_summary.to_csv(SUMMARY_FILE, index=False)
    logger.info(f"✅ Summary of top players saved to {SUMMARY_FILE}")

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
    logger.info(f"🧾 Metadata saved to {METADATA_FILE}")

    # Telegram notification
    if notify and not top_summary.empty:
        try:
            top_riser = top_summary.iloc[0]
            top_faller = merged.sort_values("impact_change").iloc[0]
            msg = (
                f"📊 Weekly Player Trends\n"
                f"Top Riser: {top_riser['PLAYER_NAME']} (+{top_riser['impact_change']:.1f} impact)\n"
                f"Top Faller: {top_faller['PLAYER_NAME']} ({top_faller['impact_change']:.1f} impact)\n"
                f"Summary saved to {SUMMARY_FILE}"
            )
            send_telegram_message(msg)
            logger.info("✅ Telegram notification sent")
        except Exception as e:
            logger.warning(f"⚠️ Failed to send Telegram notification: {e}")

    # Weekly summary
    weekly_summary = None
    if update_weekly:
        weekly_summary = build_weekly_summary(merged, notify=notify)

    return merged, top_summary, weekly_summary

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare player snapshots and generate trends")
    parser.add_argument("--no-notify", action="store_true", help="Disable Telegram notifications")
    parser.add_argument("--threshold", type=float, default=2.0, help="Impact threshold for trend classification")
    parser.add_argument("--no-weekly", action="store_true", help="Disable automatic weekly summary update")
    parser.add_argument("--fail-fast", action="store_true", help="Raise exceptions on errors instead of logging")
    args = parser.parse_args()

    compare_snapshots(
        notify=not args.no_notify,
        trend_threshold=args.threshold,
        update_weekly=not args.no_weekly,
        fail_fast=args.fail_fast
    )
