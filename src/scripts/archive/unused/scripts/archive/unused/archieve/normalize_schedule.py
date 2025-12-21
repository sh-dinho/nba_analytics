"""
normalize_schedule.py
Converts raw NBA schedule data into canonical pipeline format.
Handles date parsing issues gracefully and supports all MATCHUP formats.
"""

import pandas as pd
import re


def normalize_schedule(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw schedule DataFrame to canonical format:
    Columns: game_id, date, home_team, away_team, home_score, away_score
    """

    # ---------------------------------------------------------
    # 1. Validate required columns
    # ---------------------------------------------------------
    required = {"GAME_ID", "GAME_DATE", "TEAM_NAME", "MATCHUP", "PTS"}
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(f"normalize_schedule(): Missing required columns: {missing}")

    # ---------------------------------------------------------
    # 2. Parse dates safely
    # ---------------------------------------------------------
    df_raw["date"] = pd.to_datetime(df_raw["GAME_DATE"], errors="coerce")

    invalid_dates = df_raw[df_raw["date"].isna()]
    if not invalid_dates.empty:
        print(f"⚠️ Dropping {len(invalid_dates)} rows with invalid GAME_DATE")
        df_raw = df_raw.dropna(subset=["date"])

    # Ensure GAME_ID is string
    df_raw["GAME_ID"] = df_raw["GAME_ID"].astype(str)

    # ---------------------------------------------------------
    # 3. Robust MATCHUP parsing
    # ---------------------------------------------------------
    def parse_matchup(matchup: str, team: str):
        """
        Handles:
            'LAL vs BOS'
            'LAL vs. BOS'
            'GSW @ PHX'
            'MIL vs IND (OT)'
            'NYK vs PHI'
        """

        if not isinstance(matchup, str):
            return None, None

        # Clean punctuation and suffixes
        m = matchup.replace(".", "")
        m = re.sub(r"\(.*?\)", "", m).strip()

        if " vs " in m:
            # TEAM_NAME is home
            home = team
            away = m.split(" vs ")[1].strip()
        elif " @ " in m:
            # TEAM_NAME is away
            away = team
            home = m.split(" @ ")[1].strip()
        else:
            # Fallback: assume TEAM_NAME is home
            home = team
            away = None

        return home, away

    parsed = df_raw.apply(lambda r: parse_matchup(r["MATCHUP"], r["TEAM_NAME"]), axis=1)
    df_raw["home_team"] = parsed.apply(lambda x: x[0])
    df_raw["away_team"] = parsed.apply(lambda x: x[1])

    # ---------------------------------------------------------
    # 4. Assign scores to correct teams
    # ---------------------------------------------------------
    df_raw["home_score"] = df_raw.apply(
        lambda r: r["PTS"] if r["TEAM_NAME"] == r["home_team"] else None, axis=1
    )
    df_raw["away_score"] = df_raw.apply(
        lambda r: r["PTS"] if r["TEAM_NAME"] == r["away_team"] else None, axis=1
    )

    # ---------------------------------------------------------
    # 5. Aggregate per game_id
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 6. Final cleanup
    # ---------------------------------------------------------
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    # Drop rows missing home/away teams
    df = df.dropna(subset=["home_team", "away_team"])

    return df
