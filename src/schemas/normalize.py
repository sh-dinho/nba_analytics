# ============================================================
# File: src/schemas/normalize.py
# Purpose: Schema normalization for NBA datasets
# ============================================================

import pandas as pd


def normalize(df: pd.DataFrame, schema: str) -> pd.DataFrame:
    """
    Normalize dataframe to expected schema.
    NA-safe and production-ready.
    """

    df = df.copy()

    # -------------------------
    # Common fields
    # -------------------------
    if "GAME_ID" in df.columns:
        df["GAME_ID"] = df["GAME_ID"].astype(str)

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    # -------------------------
    # Team identifiers (NA-safe)
    # -------------------------
    for col in ["TEAM_ID", "HOME_TEAM", "AWAY_TEAM"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                "Int64"
            )  # ✅ nullable integer

    # -------------------------
    # Game outcomes
    # -------------------------
    if "PTS" in df.columns:
        df["PTS"] = pd.to_numeric(df["PTS"], errors="coerce")

    if "WL" in df.columns:
        df["WL"] = df["WL"].astype(str)

    # -------------------------
    # Schema-specific logic
    # -------------------------
    if schema == "enriched_schedule":
        # Ensure columns exist even if empty
        for col in [
            "TEAM_ID",
            "HOME_TEAM",
            "AWAY_TEAM",
            "PTS",
            "WL",
        ]:
            if col not in df.columns:
                df[col] = pd.NA

    return df
