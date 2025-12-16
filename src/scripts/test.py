import pandas as pd

df = pd.read_parquet("data/cache/historical_schedule.parquet")
print(df.head())
print(df["GAME_DATE"].dt.year.value_counts())
