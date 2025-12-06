# ============================================================
# File: scripts/ai_tracker.py
# Purpose: Track wins/losses, teams to avoid, players to watch/avoid
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import datetime

from nba_core.config import BASE_RESULTS_DIR, SUMMARY_FILE as PIPELINE_SUMMARY_FILE, log_config_snapshot
from nba_core.paths import (
    AI_TRACKER_TEAMS_FILE,
    AI_TRACKER_PLAYERS_FILE,
    AI_TRACKER_INSIGHT_FILE,
    AI_TRACKER_DASHBOARD_FILE,
    AI_TRACKER_SUMMARY_FILE,
    ensure_dirs,
)
from nba_core.log_config import init_global_logger
from notifications import send_telegram_message, send_photo

logger = init_global_logger()

ensure_dirs(strict=False)
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def update_tracker(backtest_file: Path, season="aggregate", notes="AI tracker update",
                   notify=False, avoid_threshold=0.4, watch_threshold=0.6):
    """Update AI tracker with team/player stats from a backtest file."""
    log_config_snapshot()

    if not backtest_file.exists():
        logger.warning(f"⚠️ Backtest file not found: {backtest_file}")
        return

    df = pd.read_csv(backtest_file)
    required_cols = {"team_id", "correct"}
    if not required_cols.issubset(df.columns):
        logger.error(f"❌ Missing required columns in {backtest_file}")
        return

    # Team-level aggregation
    team_stats = df.groupby("team_id").agg(
        wins=("correct", "sum"),
        total=("correct", "count")
    ).reset_index()
    team_stats["losses"] = team_stats["total"] - team_stats["wins"]
    team_stats["win_rate"] = team_stats["wins"] / team_stats["total"]
    team_stats["avoid_flag"] = team_stats["win_rate"] < avoid_threshold

    # Player-level aggregation
    player_stats = None
    if "player_id" in df.columns:
        group_cols = ["player_id"]
        if "player_name" in df.columns:
            group_cols.append("player_name")
        player_stats = df.groupby(group_cols).agg(
            wins=("correct", "sum"),
            total=("correct", "count")
        ).reset_index()
        player_stats["losses"] = player_stats["total"] - player_stats["wins"]
        player_stats["win_rate"] = player_stats["wins"] / player_stats["total"]
        player_stats["watch_flag"] = player_stats["win_rate"] > watch_threshold
        player_stats["avoid_flag"] = player_stats["win_rate"] < avoid_threshold

    # Save trackers
    team_stats.to_csv(AI_TRACKER_TEAMS_FILE, index=False)
    logger.info(f"📑 Team tracker updated → {AI_TRACKER_TEAMS_FILE}")

    if player_stats is not None:
        player_stats.to_csv(AI_TRACKER_PLAYERS_FILE, index=False)
        logger.info(f"📑 Player tracker updated → {AI_TRACKER_PLAYERS_FILE}")

    # Save insight text
    avoid_teams = team_stats.loc[team_stats["avoid_flag"], "team_id"].tolist()
    watch_players = player_stats.loc[player_stats["watch_flag"], "player_id"].tolist() if player_stats is not None else []
    avoid_players = player_stats.loc[player_stats["avoid_flag"], "player_id"].tolist() if player_stats is not None else []

    insight = (
        f"🤖 AI Tracker Insight ({season})\n"
        f"Teams to avoid: {avoid_teams}\n"
        f"Players to watch: {watch_players}\n"
        f"Players to avoid: {avoid_players}\n"
    )
    AI_TRACKER_INSIGHT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(AI_TRACKER_INSIGHT_FILE, "w") as f:
        f.write(insight)
    logger.info(f"AI insight saved → {AI_TRACKER_INSIGHT_FILE}")

    # Append to pipeline summary
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

    # Append to dedicated AI tracker summary
    try:
        AI_TRACKER_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary_row]).to_csv(
            AI_TRACKER_SUMMARY_FILE,
            mode="a",
            header=not AI_TRACKER_SUMMARY_FILE.exists(),
            index=False
        )
        logger.info(f"📈 AI tracker summary appended to {AI_TRACKER_SUMMARY_FILE}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to append AI tracker summary: {e}")

    # Notify via Telegram
    if notify:
        send_telegram_message(insight)
        logger.info("📲 AI tracker insight pushed to Telegram")


def plot_team_dashboard(team_stats: pd.DataFrame, season="aggregate") -> Path:
    """Generate a bar chart of team win rates with avoid flags highlighted."""
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

    # Annotate bars
    for x, row in zip(team_stats["team_id"].astype(str), team_stats.itertuples()):
        ax.text(x, row.win_rate + 0.02, f"{row.wins}-{row.losses}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(AI_TRACKER_DASHBOARD_FILE)
    plt.close()
    logger.info(f"📊 AI tracker dashboard saved → {AI_TRACKER_DASHBOARD_FILE}")
    return AI_TRACKER_DASHBOARD_FILE


if __name__ == "__main__":
    backtest_file = BASE_RESULTS_DIR / "backtest_results.csv"
    update_tracker(backtest_file, season="2025-2026", notes="Daily run", notify=True)

    if AI_TRACKER_TEAMS_FILE.exists():
        team_stats_df = pd.read_csv(AI_TRACKER_TEAMS_FILE)
        dashboard_img = plot_team_dashboard(team_stats_df, season="2025-2026")
        send_photo(str(dashboard_img), caption="📊 AI Tracker Team Win Rates")
