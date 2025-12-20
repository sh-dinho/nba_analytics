"""
normalize_schedule.py
Converts raw NBA schedule data into canonical pipeline format.
Handles date parsing issues gracefully.
"""

import pandas as pd


def normalize_schedule(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw schedule DataFrame to canonical format:
    Columns: game_id, date, home_team, away_team, home_score, away_score
    """
    # Ensure GAME_DATE exists
    if "GAME_DATE" not in df_raw.columns:
        raise ValueError("Input DataFrame missing 'GAME_DATE' column")

    # Parse dates safely
    df_raw["date"] = pd.to_datetime(df_raw["GAME_DATE"], errors="coerce")

    # Drop rows with invalid dates
    invalid_dates = df_raw[df_raw["date"].isna()]
    if not invalid_dates.empty:
        print(f"⚠️ Dropping {len(invalid_dates)} rows with invalid dates")
        df_raw = df_raw.dropna(subset=["date"])

    # Ensure GAME_ID is string
    df_raw["GAME_ID"] = df_raw["GAME_ID"].astype(str)

    # Extract home and away teams from MATCHUP
    def parse_matchup(row):
        matchup = row.get("MATCHUP", "")
        team = row.get("TEAM_NAME")
        if " vs. " in matchup:
            home, away = matchup.split(" vs. ")
        elif " @ " in matchup:
            away, home = matchup.split(" @ ")
        else:
            home = team
            away = None
        return pd.Series([home, away])

    df_raw[["home_team", "away_team"]] = df_raw.apply(parse_matchup, axis=1)

    # Assign scores to correct teams
    df_raw["home_score"] = df_raw.apply(
        lambda r: r["PTS"] if r["TEAM_NAME"] == r["home_team"] else None, axis=1
    )
    df_raw["away_score"] = df_raw.apply(
        lambda r: r["PTS"] if r["TEAM_NAME"] == r["away_team"] else None, axis=1
    )

    # Aggregate per game
    df = (
        df_raw.groupby("GAME_ID")
        .agg(
            date=("date", "first"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            home_score=("home_score", "max"),
            away_score=("away_score", "max"),
        )
        .reset_index()
    )

    # Ensure numeric scores
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    # Drop rows with missing home/away teams
    df = df.dropna(subset=["home_team", "away_team"])

    return df
