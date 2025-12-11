# ============================================================
# File: src/tracker/game_tracker.py
# Purpose: Build tracker with season + feature transparency
# Project: nba_analysis
# Version: 2.4 (adds export functions for CSV/Parquet + summary logging)
#
# Dependencies:
# - pandas
# - logging (standard library)
# ============================================================

import pandas as pd
import logging
import os


def build_game_tracker(features_df, predictions_df, player_info_df, used_features=None):
    """
    Builds a game tracker by merging the features, predictions, and player information.

    Arguments:
    features_df -- DataFrame containing the features for the games
    predictions_df -- DataFrame containing the predictions for the games
    player_info_df -- DataFrame containing player information for the games
    used_features -- List of features used in the training model (optional)

    Returns:
    DataFrame -- A game tracker with relevant information and recommendations
    """

    # Defensive copy to avoid mutating inputs
    features_df = features_df.copy()
    predictions_df = predictions_df.copy()
    player_info_df = player_info_df.copy()

    # Merge features with predictions and player info
    tracker = pd.merge(features_df, predictions_df, on="GAME_ID", how="left")
    tracker = pd.merge(tracker, player_info_df, on=["GAME_ID", "TEAM_ID"], how="left")

    # Add recommendation based on prediction confidence
    if "prediction_confidence" in tracker.columns:
        tracker["Recommendation"] = pd.cut(
            tracker["prediction_confidence"],
            bins=[-float("inf"), 0.55, 0.75, float("inf")],
            labels=["Avoid", "Watch", "Stake"],
            right=False
        ).fillna("Unknown")
    else:
        tracker["Recommendation"] = "Unknown"

    # Add transparency: which features were used in training
    tracker["FeaturesUsed"] = ", ".join(used_features) if used_features else None

    # Ensure all expected columns exist
    expected_cols = [
        "Season", "GAME_ID", "Date", "HomeTeam", "AwayTeam",
        "PlayerNames", "Recommendation", "FeaturesUsed"
    ]

    for col in expected_cols:
        if col not in tracker.columns:
            tracker[col] = None

    # Reorder columns to match the expected order
    tracker = tracker[expected_cols]

    # -----------------------------
    # SUMMARY LOGGING
    # -----------------------------
    logging.info(f"Game tracker built with {len(tracker)} games.")
    logging.info("Recommendation distribution:")
    logging.info(tracker["Recommendation"].value_counts(dropna=False).to_string())

    return tracker


def save_tracker(tracker: pd.DataFrame, path: str, fmt: str = "parquet"):
    """
    Save the tracker DataFrame to disk in the specified format.

    Arguments:
    tracker -- DataFrame to save
    path -- Output file path
    fmt -- Format to save ('parquet' or 'csv')
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        if fmt == "parquet":
            tracker.to_parquet(path, index=False)
        elif fmt == "csv":
            tracker.to_csv(path, index=False)
        else:
            raise ValueError("Unsupported format. Use 'parquet' or 'csv'.")
        logging.info(f"Tracker successfully saved to {path} ({fmt.upper()})")
    except Exception as e:
        logging.error(f"Error saving tracker: {e}")
