# ============================================================
# Path: src/analytics/rankings.py
# Purpose: Track top teams/players, betting recommendations,
#          and winning streaks from prediction history + box scores
# ============================================================

import pandas as pd
import os

def compute_team_rankings(history_file, out_dir="data/analytics"):
    """
    Compute top 6 performing teams per conference.
    Metrics: win %, prediction accuracy, avg predicted probability.
    """
    df = pd.read_parquet(history_file)

    # Precompute correct predictions
    df["correct_pred"] = (df["pred_label"] == df["win"]).astype(int)

    # Aggregate team stats
    team_stats = df.groupby("TEAM_ID").agg(
        games_played=("GAME_ID", "count"),
        wins=("win", "sum"),
        avg_pred_proba=("pred_proba", "mean"),
        pred_accuracy=("correct_pred", "mean")
    ).reset_index()

    team_stats["win_pct"] = team_stats["wins"] / team_stats["games_played"]

    # Add conference info (requires mapping TEAM_ID → conference)
    conf_map = pd.read_csv("data/raw/team_conference_map.csv")  # TEAM_ID, Conference
    team_stats = team_stats.merge(conf_map, on="TEAM_ID", how="left")
    team_stats["Conference"] = team_stats["Conference"].fillna("Unknown")

    # Top 6 per conference
    top_teams = team_stats.sort_values(["Conference", "win_pct"], ascending=[True, False])
    top_teams = top_teams.groupby("Conference").head(6)

    os.makedirs(out_dir, exist_ok=True)
    top_teams.to_parquet(os.path.join(out_dir, "team_rankings.parquet"), index=False)
    top_teams.to_csv(os.path.join(out_dir, "team_rankings.csv"), index=False)

    return top_teams


def compute_player_rankings(player_box_file, out_dir="data/analytics"):
    """
    Compute top 6 performing players per conference.
    Metrics: avg points, consistency (lower std = more consistent), prop hit rate (20+ pts).
    """
    df = pd.read_parquet(player_box_file)

    player_stats = df.groupby("PLAYER_ID").agg(
        avg_points=("PTS", "mean"),
        games_played=("GAME_ID", "count"),
        prop_hit_rate=("PTS", lambda x: (x >= 20).mean()),
        points_std=("PTS", "std")
    ).reset_index()

    # Add conference info (PLAYER_ID → TEAM_ID → Conference)
    team_map = pd.read_csv("data/raw/team_conference_map.csv")
    player_team_map = df[["PLAYER_ID", "TEAM_ID"]].drop_duplicates()
    player_stats = player_stats.merge(player_team_map, on="PLAYER_ID", how="left")
    player_stats = player_stats.merge(team_map, on="TEAM_ID", how="left")
    player_stats["Conference"] = player_stats["Conference"].fillna("Unknown")

    # Top 6 per conference
    top_players = player_stats.sort_values(["Conference", "avg_points"], ascending=[True, False])
    top_players = top_players.groupby("Conference").head(6)

    os.makedirs(out_dir, exist_ok=True)
    top_players.to_parquet(os.path.join(out_dir, "player_rankings.parquet"), index=False)
    top_players.to_csv(os.path.join(out_dir, "player_rankings.csv"), index=False)

    return top_players


def betting_recommendations(team_rankings, threshold=0.55):
    """
    Generate teams to bet on vs. avoid.
    Bet On: win_pct > threshold and pred_accuracy > threshold
    Avoid: win_pct < threshold or pred_accuracy < threshold
    Returns DataFrames instead of just IDs for richer context.
    """
    bet_on = team_rankings[
        (team_rankings["win_pct"] > threshold) &
        (team_rankings["pred_accuracy"] > threshold)
    ]

    avoid = team_rankings[
        (team_rankings["win_pct"] < threshold) |
        (team_rankings["pred_accuracy"] < threshold)
    ]

    return {"bet_on": bet_on, "avoid": avoid}


def track_winning_streaks(history_file, out_dir="data/analytics"):
    """
    Track team winning streaks from history.
    Records both max streak and current streak.
    """
    df = pd.read_parquet(history_file)
    streaks = []

    for team_id, group in df.groupby("TEAM_ID"):
        group = group.sort_values("prediction_date")
        current_streak = 0
        max_streak = 0
        latest_streak = 0
        for win in group["win"]:
            if win == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
                latest_streak = current_streak
            else:
                current_streak = 0
        streaks.append({"TEAM_ID": team_id, "max_streak": max_streak, "current_streak": latest_streak})

    streaks_df = pd.DataFrame(streaks)
    os.makedirs(out_dir, exist_ok=True)
    streaks_df.to_parquet(os.path.join(out_dir, "winning_streaks.parquet"), index=False)
    streaks_df.to_csv(os.path.join(out_dir, "winning_streaks.csv"), index=False)

    return streaks_df
