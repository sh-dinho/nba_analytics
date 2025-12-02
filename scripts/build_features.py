# File: scripts/build_features.py

import os
import pandas as pd
import numpy as np
from scripts.utils import setup_logger
import datetime

logger = setup_logger("build_features")
DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES_FILE = os.path.join(DATA_DIR, "training_features.csv")
SUMMARY_LOG = os.path.join(RESULTS_DIR, "features_summary.csv")

def main():
    stats_file = os.path.join(DATA_DIR, "player_stats.csv")

    if not os.path.exists(stats_file):
        logger.error(f"{stats_file} not found. Fetch player stats first.")
        raise FileNotFoundError(stats_file)

    logger.info("Building training features...")

    # Load raw player stats
    df = pd.read_csv(stats_file)

    # Aggregate stats per team
    team_stats = df.groupby("TEAM_ABBREVIATION").agg({
        "PTS": "mean",
        "AST": "mean",
        "REB": "mean",
        "STL": "mean",
        "BLK": "mean"
    }).reset_index()

    # Create dummy matchups
    games = []
    teams = team_stats["TEAM_ABBREVIATION"].tolist()
    np.random.shuffle(teams)  # randomize to increase diversity
    game_id = 1
    for i in range(0, len(teams), 2):
        if i + 1 >= len(teams):
            break
        home = teams[i]
        away = teams[i + 1]
        home_stats = team_stats[team_stats["TEAM_ABBREVIATION"] == home].iloc[0]
        away_stats = team_stats[team_stats["TEAM_ABBREVIATION"] == away].iloc[0]

        features = {
            "game_id": game_id,
            "home_team": home,
            "away_team": away,
            "home_pts_avg": home_stats["PTS"],
            "away_pts_avg": away_stats["PTS"],
            "home_ast_avg": home_stats["AST"],
            "away_ast_avg": away_stats["AST"],
            "home_reb_avg": home_stats["REB"],
            "away_reb_avg": away_stats["REB"],
            "home_win": 1 if home_stats["PTS"] > away_stats["PTS"] else 0
        }
        games.append(features)
        game_id += 1

    features_df = pd.DataFrame(games)

    # 🔒 Safeguard: ensure label diversity
    if features_df["home_win"].nunique() == 1:
        logger.warning("⚠️ Only one label found in home_win. Injecting synthetic diversity.")
        flip_idx = np.random.choice(features_df.index, size=len(features_df)//2, replace=False)
        features_df.loc[flip_idx, "home_win"] = 1 - features_df.loc[flip_idx, "home_win"]

    # Save clean features file
    features_df.to_csv(FEATURES_FILE, index=False)
    logger.info(f"✅ Training features saved to {FEATURES_FILE} | Games built: {len(features_df)}")

    # Summary stats
    avg_pts_diff = (features_df["home_pts_avg"] - features_df["away_pts_avg"]).mean()
    avg_ast_diff = (features_df["home_ast_avg"] - features_df["away_ast_avg"]).mean()
    avg_reb_diff = (features_df["home_reb_avg"] - features_df["away_reb_avg"]).mean()

    logger.info(
        f"📊 Feature summary: Avg PTS diff={avg_pts_diff:.2f}, "
        f"Avg AST diff={avg_ast_diff:.2f}, Avg REB diff={avg_reb_diff:.2f}"
    )

    # Append summary to rolling log
    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_entry = pd.DataFrame([{
        "timestamp": run_time,
        "games_built": len(features_df),
        "avg_pts_diff": avg_pts_diff,
        "avg_ast_diff": avg_ast_diff,
        "avg_reb_diff": avg_reb_diff
    }])

    if os.path.exists(SUMMARY_LOG):
        summary_entry.to_csv(SUMMARY_LOG, mode="a", header=False, index=False)
    else:
        summary_entry.to_csv(SUMMARY_LOG, index=False)

    logger.info(f"📁 Features summary appended to {SUMMARY_LOG}")

    return features_df

if __name__ == "__main__":
    main()