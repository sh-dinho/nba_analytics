import pandas as pd
from loguru import logger
from src.config.paths import SCHEDULE_SNAPSHOT, LONG_SNAPSHOT


def check_data():
    if not SCHEDULE_SNAPSHOT.exists():
        logger.error("No snapshot found.")
        return

    df = pd.read_parquet(SCHEDULE_SNAPSHOT)

    print("\n=== DATA REPORT ===")
    print(f"Total Rows: {len(df)}")
    print(f"Columns Found: {list(df.columns)}")

    # Check for scores
    score_cols = ["score_home", "score_away"]
    games_with_scores = df[df[score_cols].notna().all(axis=1)].shape[0]
    print(f"Games with Scores: {games_with_scores}")

    # Safely show recent games
    cols_to_show = [
        c
        for c in [
            "date",
            "home_team",
            "away_team",
            "score_home",
            "score_away",
            "status",
        ]
        if c in df.columns
    ]

    if "date" in df.columns:
        print("\n--- Latest 5 Entries by Date ---")
        print(df.sort_values("date").tail(5)[cols_to_show])

        # Freshness check
        last_date = pd.to_datetime(df["date"]).max().date()
        days_ago = (pd.Timestamp.today().date() - last_date).days
        print(f"\n⏱️ Last game was {days_ago} days ago ({last_date})")
    else:
        print("\n⚠️ No 'date' column found, showing last 5 entries by index instead:")
        print(df.tail(5)[cols_to_show])


if __name__ == "__main__":
    check_data()
