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
    games_with_scores = df[df["home_score"].notna()].shape[0]
    print(f"Games with Scores: {games_with_scores}")

    # Safely show recent games
    cols_to_show = [
        c
        for c in ["date", "home_team", "away_team", "home_score", "status"]
        if c in df.columns
    ]
    print("\n--- Latest 5 Entries ---")
    print(df.sort_values("date").tail(5)[cols_to_show])


check_data()


if __name__ == "__main__":
    check_data()
# from src.ingestion.pipeline import IngestionPipeline
# from src.features.builder import FeatureBuilder
# from src.features.feature_store import FeatureStore
#
# # 1. Transform Wide to Long
# pipeline = IngestionPipeline()
# df_long = pipeline.load_long_format()
#
# # 2. Build Rolling Features (Win rates, etc.)
# fb = FeatureBuilder(window=10)
# features_df = fb.build(df_long)
#
# # 3. Save to Store
# fs = FeatureStore()
# fs.save_snapshot(features_df, kind="training")

# import pandas as pd
# from datetime import date
# from src.config.paths import SCHEDULE_SNAPSHOT
#
# df = pd.read_parquet(SCHEDULE_SNAPSHOT)
# today = date.today()
#
# print(f"Checking for date: {today} (Type: {type(today)})")
# print(f"Dates in file: {df['date'].unique()}")
# print(f"First date type in file: {type(df['date'].iloc[0])}")
#
# todays_games = df[df["date"] == today]
# print(f"\nGames found for today: {len(todays_games)}")
# print(todays_games)
