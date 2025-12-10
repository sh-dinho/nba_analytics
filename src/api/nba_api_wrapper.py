# ============================================================
# File: src/api/nba_api_wrapper.py
# Purpose: Wrap nba_api endpoints for fetching NBA game data
# Project: nba_analysis
# Version: 1.0
# ============================================================

import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder

# -----------------------------
# CONFIG
# -----------------------------
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]  # historical + current seasons

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def fetch_season_games(season: str) -> pd.DataFrame:
    """
    Fetch all games for a given season using nba_api.
    Returns a DataFrame with columns: GAME_DATE, TEAM_NAME, MATCHUP
    """
    print(f"[INFO] Fetching games for season {season}...")
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        df = gamefinder.get_data_frames()[0]
        df = df[['GAME_DATE', 'TEAM_NAME', 'MATCHUP']]
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        return df
    except Exception as e:
        print(f"[ERROR] Failed to fetch season {season}: {e}")
        return pd.DataFrame()


def fetch_today_games() -> pd.DataFrame:
    """
    Fetch today's NBA games based on the latest season.
    Returns a DataFrame with columns: GAME_DATE, TEAM_NAME, MATCHUP, home_team, away_team
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        current_season = SEASONS[-1]
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=current_season)
        games = gamefinder.get_data_frames()[0]
    except Exception as e:
        print(f"[ERROR] Failed to fetch today's games: {e}")
        return pd.DataFrame()

    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    today_games = games[games['GAME_DATE'] == pd.to_datetime(today_str)]
    if today_games.empty:
        return pd.DataFrame()

    # Extract home and away teams from MATCHUP
    today_games['home_team'] = today_games['MATCHUP'].apply(
        lambda x: x.split(' vs. ')[-1] if 'vs.' in x else None
    )
    today_games['away_team'] = today_games['MATCHUP'].apply(
        lambda x: x.split(' @ ')[0] if ' @ ' in x else None
    )

    return today_games.dropna(subset=['home_team', 'away_team'])


def update_historical_games(existing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch and combine historical games for all seasons.
    Removes duplicates based on GAME_DATE, TEAM_NAME, and MATCHUP.
    """
    all_new_games = []
    for season in SEASONS:
        new_games = fetch_season_games(season)
        if not new_games.empty:
            all_new_games.append(new_games)

    if all_new_games:
        combined_df = pd.concat(all_new_games, ignore_index=True)
        combined_df = pd.concat([existing_df, combined_df], ignore_index=True).drop_duplicates(
            subset=['GAME_DATE', 'TEAM_NAME', 'MATCHUP']
        )
    else:
        combined_df = existing_df

    print(f"[INFO] Historical games updated. Total games: {len(combined_df)}")
    return combined_df
