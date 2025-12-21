"""
NBA Schedule Verification Script (Canonical Snapshot Version)

Verifies:
- Game counts per season
- Duplicate game IDs
- Regular Season completeness (1230 games)
- Playoff completeness (64–127 games)
- Freshness (last game date)
- Snapshot integrity

Reads from:
    data/snapshots/schedule_snapshot.parquet
"""

import pandas as pd
from pathlib import Path

SNAPSHOT_PATH = Path("data/canonical/schedule.parquet")

# ---------------------------------------------------------
#  UTILITIES
# ---------------------------------------------------------


def season_from_date(d: pd.Timestamp) -> str:
    year = d.year
    if d.month >= 10:
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        return f"{year - 1}-{str(year)[-2:]}"


def load_snapshot():
    """Load canonical schedule snapshot."""
    if not SNAPSHOT_PATH.exists():
        print(f"❌ Snapshot not found: {SNAPSHOT_PATH}")
        return None

    try:
        df = pd.read_parquet(SNAPSHOT_PATH)
        if df.empty:
            print("⚠️ Snapshot exists but is EMPTY.")
            return None

        print(f"📦 Loaded snapshot → {SNAPSHOT_PATH}")
        return df

    except Exception as e:
        print(f"⚠️ Failed to read snapshot: {e}")
        return None


# ---------------------------------------------------------
#  VERIFICATION
# ---------------------------------------------------------


def verify_data():
    df = load_snapshot()
    if df is None:
        return

    # Defensive normalization
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["game_id"] = df["game_id"].astype(str)

    # Derive season
    df["season"] = df["date"].apply(season_from_date)

    # Count unique games per season
    counts = df.groupby("season")["game_id"].nunique().sort_index()

    print("\n📊 Game counts per season:")
    for season, count in counts.items():
        print(f"  {season}: {count} games")

    # -----------------------------------------------------
    #  DUPLICATE GAME ID CHECK
    # -----------------------------------------------------

    print("\n🔍 Checking for duplicate game IDs...")

    dupes = df[df["game_id"].duplicated(keep=False)]

    if dupes.empty:
        print("  ✅ No duplicate game IDs found.")
    else:
        print(f"  ❌ Found {dupes['game_id'].nunique()} duplicated game IDs!")
        for gid in dupes["game_id"].unique():
            print(f"    - {gid}")

    # -----------------------------------------------------
    #  SEASON COMPLETENESS CHECK
    # -----------------------------------------------------

    print("\n🔍 Checking season completeness...")

    EXPECTED_REGULAR = 1230
    PLAYOFF_MIN = 64
    PLAYOFF_MAX = 127

    for season in counts.index:
        season_df = df[df["season"] == season]

        regular_games = season_df[season_df["season_type"] == "Regular Season"][
            "game_id"
        ].nunique()
        playoff_games = season_df[season_df["season_type"] == "Playoffs"][
            "game_id"
        ].nunique()

        print(f"\nSeason {season}:")
        print(f"  Regular Season: {regular_games} games (expected {EXPECTED_REGULAR})")
        print(
            f"  Playoffs:       {playoff_games} games (expected {PLAYOFF_MIN}–{PLAYOFF_MAX})"
        )

        if regular_games != EXPECTED_REGULAR:
            print("  ⚠️ Regular Season appears incomplete.")

        if not (PLAYOFF_MIN <= playoff_games <= PLAYOFF_MAX):
            print("  ⚠️ Playoffs appear incomplete or missing.")

    # -----------------------------------------------------
    #  FRESHNESS CHECK
    # -----------------------------------------------------

    last_game_date = df["date"].max().date()
    today = pd.Timestamp.today().date()

    print("\n⏱️ Freshness Check:")
    print(f"  Last recorded game: {last_game_date}")

    if last_game_date >= today or last_game_date >= today - pd.Timedelta(days=1):
        print("  ✅ Data appears up to date.")
    else:
        print("  ⚠️ Data may be outdated.")


# ---------------------------------------------------------
#  MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    verify_data()
