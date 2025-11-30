import sqlite3
import pandas as pd
import yaml
import logging

logging.basicConfig(level=logging.INFO)

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)
DB_PATH = CONFIG["database"]["path"]

def connect():
    return sqlite3.connect(DB_PATH)

def fetch_historical_games(season: int) -> pd.DataFrame:
    with connect() as con:
        df = pd.read_sql(
            "SELECT * FROM nba_games WHERE season=? ORDER BY date ASC",
            con,
            params=(season,)
        )
    return df

# --- THIS FUNCTION MUST EXIST ---
def engineer_features(df_games: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for model training:
    - home_win (classification target)
    - total_points (regression target)
    - point_diff, rolling averages, Elo ratings
    """
    if df_games.empty:
        return pd.DataFrame()

    df = df_games.copy()

    # Targets
    df["home_win"] = (df["winner"] == df["home_team"]).astype(int)
    df["total_points"] = df["home_score"] + df["away_score"]

    # Basic feature
    df["point_diff"] = df["home_score"] - df["away_score"]

    # Rolling averages
    def rolling_stats(team_col, score_col, opp_score_col, prefix):
        stats = []
        team_groups = df.groupby(team_col)
        for team, group in team_groups:
            group = group.sort_values("date")
            avg_scored = group[score_col].expanding().mean().shift(1)
            avg_allowed = group[opp_score_col].expanding().mean().shift(1)
            stats.append(pd.DataFrame({
                "game_id": group["game_id"],
                f"{prefix}_avg_scored": avg_scored,
                f"{prefix}_avg_allowed": avg_allowed
            }))
        return pd.concat(stats)

    home_stats = rolling_stats("home_team", "home_score", "away_score", "home")
    away_stats = rolling_stats("away_team", "away_score", "home_score", "away")

    df = df.merge(home_stats, on="game_id", how="left")
    df = df.merge(away_stats, on="game_id", how="left")

    # Elo ratings
    ratings = {}
    elo_home, elo_away = [], []
    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        ratings.setdefault(home, 1500)
        ratings.setdefault(away, 1500)

        home_rating, away_rating = ratings[home], ratings[away]
        elo_home.append(home_rating)
        elo_away.append(away_rating)

        # Update ratings
        if row["winner"] == home:
            score_home = 1
        elif row["winner"] == away:
            score_home = 0
        else:
            score_home = 0.5

        expected_home = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
        new_home = home_rating + 20 * (score_home - expected_home)
        new_away = away_rating + 20 * ((1 - score_home) - (1 - expected_home))
        ratings[home], ratings[away] = new_home, new_away

    df["home_elo"] = elo_home
    df["away_elo"] = elo_away

    df.fillna(0, inplace=True)

    feature_cols = [
        "point_diff",
        "home_avg_scored", "home_avg_allowed",
        "away_avg_scored", "away_avg_allowed",
        "home_elo", "away_elo"
    ]
    target_cols = ["home_win", "total_points"]

    return df[feature_cols + target_cols]