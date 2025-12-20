"""
Build completed_games.parquet for the last 5 NBA seasons.
Uses LeagueGameLog (correct endpoint for historical games).
"""

import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints import LeagueGameLog

OUTPUT_PATH = Path("data/raw/completed_games.parquet")

SEASONS = [
    "2024-25",
    "2023-24",
    "2022-23",
    "2021-22",
    "2020-21",
]


def build_completed_games():
    print("Fetching completed NBA games for last 5 seasons...")

    all_games = []

    for season in SEASONS:
        print(f"  → Fetching season {season}")

        log = LeagueGameLog(season=season, season_type_all_star="Regular Season")

        df = log.get_data_frames()[0]

        # Normalize date
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

        # Split into home and away rows
        df_home = df[df["MATCHUP"].str.contains(" vs ", na=False)].copy()
        df_away = df[df["MATCHUP"].str.contains(" @ ", na=False)].copy()

        # Extract team names
        df_home["home_team"] = df_home["TEAM_NAME"]
        df_home["away_team"] = df_home["MATCHUP"].str.split(" vs ").str[1]

        df_away["away_team"] = df_away["TEAM_NAME"]
        df_away["home_team"] = df_away["MATCHUP"].str.split(" @ ").str[1]

        # Merge home + away rows
        merged = pd.merge(
            df_home[["GAME_ID", "GAME_DATE", "home_team", "PTS"]],
            df_away[["GAME_ID", "away_team", "PTS"]],
            on="GAME_ID",
            suffixes=("_home", "_away"),
        )

        merged = merged.rename(
            columns={
                "PTS_home": "home_score",
                "PTS_away": "away_score",
                "GAME_DATE": "date",
            }
        )

        all_games.append(merged)

    df_all = pd.concat(all_games, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved completed games → {OUTPUT_PATH}")
    print(f"Total games: {len(df_all)}")

    return OUTPUT_PATH


if __name__ == "__main__":
    build_completed_games()
