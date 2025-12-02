import os
import pandas as pd
from scripts.utils import setup_logger

logger = setup_logger("build_features")
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def main():
    """
    Build training features from player stats.
    Generates 'training_features.csv' containing game-level features and target.
    """
    stats_file = os.path.join(DATA_DIR, "player_stats.csv")
    features_file = os.path.join(DATA_DIR, "training_features.csv")

    if not os.path.exists(stats_file):
        logger.error(f"{stats_file} not found. Fetch player stats first.")
        raise FileNotFoundError(stats_file)

    logger.info("Building training features...")

    # Load raw player stats
    df = pd.read_csv(stats_file)

    # Example: Aggregate stats per team
    team_stats = df.groupby("TEAM_ABBREVIATION").agg({
        "PTS": "mean",
        "AST": "mean",
        "REB": "mean",
        "STL": "mean",
        "BLK": "mean"
    }).reset_index()

    # For simplicity, create dummy matchups
    games = []
    teams = team_stats["TEAM_ABBREVIATION"].tolist()
    game_id = 1
    for i in range(0, len(teams), 2):
        if i+1 >= len(teams):
            break
        home = teams[i]
        away = teams[i+1]
        home_stats = team_stats[team_stats["TEAM_ABBREVIATION"]==home].iloc[0]
        away_stats = team_stats[team_stats["TEAM_ABBREVIATION"]==away].iloc[0]

        # Features: difference in stats
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
            "home_win": 1 if home_stats["PTS"] > away_stats["PTS"] else 0  # target
        }
        games.append(features)
        game_id += 1

    features_df = pd.DataFrame(games)
    features_df.to_csv(features_file, index=False)
    logger.info(f"Training features saved to {features_file}")

    return features_df

if __name__ == "__main__":
    main()
