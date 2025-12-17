import pandas as pd

# List of the season parquet files
season_files = [
    "data/history/season_2022-23.parquet",
    "data/history/season_2023-24.parquet",
    "data/history/season_2024-25.parquet",
    "data/history/season_2025-26.parquet",
]

# Read and concatenate the DataFrames
dfs = [pd.read_parquet(file) for file in season_files]
historical_schedule = pd.concat(dfs, ignore_index=True)

# Save the combined DataFrame as a new parquet file
historical_schedule.to_parquet("data/history/historical_schedule.parquet")

print("Historical schedule parquet file created successfully!")
