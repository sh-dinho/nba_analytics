# ============================================================
# Path: src/tracker/game_tracker.py
# Purpose: Build tracker with season + feature transparency
# Version: 2.1 (robust column handling + safe merges)
# ============================================================

import pandas as pd

def build_game_tracker(features_df, predictions_df, player_info_df, used_features=None):
    # Merge features with predictions and player info
    tracker = features_df.merge(predictions_df, on="GAME_ID", how="left")
    tracker = tracker.merge(player_info_df, on=["GAME_ID", "TEAM_ID"], how="left")

    def classify(row):
        conf = row.get("prediction_confidence", None)
        if conf is None:
            return "Unknown"
        if conf >= 0.75:
            return "Stake"
        elif conf < 0.55:
            return "Avoid"
        else:
            return "Watch"

    tracker["Recommendation"] = tracker.apply(classify, axis=1)

    # Add transparency: which features were used in training
    if used_features is not None:
        tracker["FeaturesUsed"] = ", ".join(used_features)
    else:
        tracker["FeaturesUsed"] = None

    # Ensure all expected columns exist
    expected_cols = ["Season", "GAME_ID", "Date", "HomeTeam", "AwayTeam", "PlayerNames", "Recommendation", "FeaturesUsed"]
    for col in expected_cols:
        if col not in tracker.columns:
            tracker[col] = None

    return tracker[expected_cols]
