# ============================================================
# File: scripts/ai_tracker.py
# Purpose: Track wins/losses, teams to avoid, players to watch/avoid
# ============================================================

import pandas as pd
from pathlib import Path
from core.config import BASE_RESULTS_DIR, SUMMARY_FILE as PIPELINE_SUMMARY_FILE
from core.log_config import init_global_logger
from notifications import send_message
import matplotlib.pyplot as plt

logger = init_global_logger()

TRACKER_FILE = BASE_RESULTS_DIR / "ai_tracker.csv"
INSIGHT_FILE = BASE_RESULTS_DIR / "ai_tracker_insight.txt"

def update_tracker(backtest_file: Path, season="aggregate", notes="AI tracker update", notify=False):
    if not backtest_file.exists():
        logger.warning(f"⚠️ Backtest file not found: {backtest_file}")
        return

    df = pd.read_csv(backtest_file)

    # Team-level aggregation
    team_stats = df.groupby("team_id").agg(
        wins=("correct", "sum"),
        total=("correct", "count")
    ).reset_index()
    team_stats["losses"] = team_stats["total"] - team_stats["wins"]
    team_stats["win_rate"] = team_stats["wins"] / team_stats["total"]
    team_stats["avoid_flag"] = team_stats["win_rate"] < 0.4

    # Player-level aggregation (if player_id present)
    player_stats = None
    if "player_id" in df.columns:
        player_stats = df.groupby("player_id").agg(
            wins=("correct", "sum"),
            total=("correct", "count")
        ).reset_index()
        player_stats["losses"] = player_stats["total"] - player_stats["wins"]
        player_stats["win_rate"] = player_stats["wins"] / player_stats["total"]
        player_stats["watch_flag"] = player_stats["win_rate"] > 0.6
        player_stats["avoid_flag"] = player_stats["win_rate"] < 0.4

    # Save tracker
    tracker_out = {"teams": team_stats}
    if player_stats is not None:
        tracker_out["players"] = player_stats

    pd.concat(tracker_out.values(), axis=0).to_csv(TRACKER_FILE, index=False)
    logger.info(f"📑 AI tracker updated → {TRACKER_FILE}")

    # Save insight
    avoid_teams = team_stats.loc[team_stats["avoid_flag"], "team_id"].tolist()
    watch_players = player_stats.loc[player_stats["watch_flag"], "player_id"].tolist() if player_stats is not None else []
    avoid_players = player_stats.loc[player_stats["avoid_flag"], "player_id"].tolist() if player_stats is not None else []

    insight = (
        f"🤖 AI Tracker Insight ({season})\n"
        f"Teams to avoid: {avoid_teams}\n"
        f"Players to watch: {watch_players}\n"
        f"Players to avoid: {avoid_players}\n"
    )
    with open(INSIGHT_FILE, "w") as f:
        f.write(insight)
    logger.info(f"AI insight saved → {INSIGHT_FILE}")

    # Append to pipeline_summary
    run_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_row = {
        "timestamp": run_time,
        "season": season,
        "target": "ai_tracker",
        "notes": notes,
        "num_avoid_teams": len(avoid_teams),
        "num_watch_players": len(watch_players),
        "num_avoid_players": len(avoid_players),
    }
    pd.DataFrame([summary_row]).to_csv(
        PIPELINE_SUMMARY_FILE,
        mode="a",
        header=not Path(PIPELINE_SUMMARY_FILE).exists(),
        index=False
    )

    if notify:
        send_message(insight)
        logger.info("📲 AI tracker insight pushed to Telegram")


def plot_team_dashboard(team_stats, season="aggregate"):
    """
    Generate a bar chart of team win rates with avoid flags highlighted.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = team_stats["win_rate"].apply(
        lambda x: "green" if x >= 0.6 else ("orange" if x >= 0.4 else "red")
    )

    ax.bar(team_stats["team_id"].astype(str), team_stats["win_rate"], color=colors)
    ax.set_title(f"AI Tracker — Team Win Rates ({season})")
    ax.set_xlabel("Team ID")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Annotate with wins/losses
    for idx, row in team_stats.iterrows():
        ax.text(idx, row["win_rate"] + 0.02,
                f"{row['wins']}-{row['losses']}",
                ha="center", fontsize=8)

    plt.tight_layout()
    dashboard_path = BASE_RESULTS_DIR / "ai_tracker_dashboard.png"
    plt.savefig(dashboard_path)
    plt.close()
    logger.info(f"📊 AI tracker dashboard saved → {dashboard_path}")
    return dashboard_path

    dashboard_path = plot_team_dashboard(team_stats, season=season)

    if notify:
        send_message(f"📊 AI Tracker Dashboard ({season})")
        send_photo(str(dashboard_path), caption="Team Win Rates — Avoid flags highlighted")
