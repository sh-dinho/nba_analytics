import pandas as pd

# Load the features dataframe from the generated parquet file
features_df = pd.read_parquet("data/cache/features_full.parquet")

# Ensure GAME_DATE is in datetime format
features_df["GAME_DATE"] = pd.to_datetime(features_df["GAME_DATE"])

# If TARGET column doesn't exist, create it based on scores (example for home win/loss)
if "TARGET" not in features_df.columns:
    features_df["TARGET"] = (
        features_df["HOME_SCORE"] > features_df["AWAY_SCORE"]
    ).astype(int)

# Calculate RestDays (difference between consecutive games for the same team)
features_df["RestDays"] = features_df.groupby("TEAM_NAME")["GAME_DATE"].diff().dt.days

# Calculate cumulative win percentage for the team and opponent
features_df["TeamWinPctToDate"] = (
    features_df.groupby("TEAM_NAME")["TARGET"]
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)
)
features_df["OppWinPctToDate"] = (
    features_df.groupby("OPPONENT_TEAM_ID")["TARGET"]
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)
)

# Handle missing values (if needed)
features_df["RestDays"] = features_df["RestDays"].fillna(0)  # or other handling method
features_df["TeamWinPctToDate"] = features_df["TeamWinPctToDate"].fillna(
    0
)  # or other handling method
features_df["OppWinPctToDate"] = features_df["OppWinPctToDate"].fillna(
    0
)  # or other handling method

# Save the cleaned data back to a Parquet or CSV file
features_df.to_parquet("data/cache/features_full_cleaned.parquet")
features_df.to_csv("data/csv/features_full_cleaned.csv", index=False)
