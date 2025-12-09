# ============================================================
# Path: src/tracker/game_tracker.py
# Purpose: Build tracker with season + feature transparency
# Version: 2.0
# ============================================================

import pandas as pd

def build_game_tracker(features_df, predictions_df, player_info_df, used_features=None):
    tracker = features_df.merge(predictions_df, on="GAME_ID", how="left")
    tracker = tracker.merge(player_info_df, on=["GAME_ID", "TEAM_ID"], how="left")

    def classify(row):
        if row["prediction_confidence"] >= 0.75:
            return "Stake"
        elif row["prediction_confidence"] <= 0.55:
            return "Avoid"
        else:
            return "Watch"

    tracker["Recommendation"] = tracker.apply(classify, axis=1)

    # Add transparency: which features were used in training
    if used_features is not None:
        tracker["FeaturesUsed"] = ", ".join(used_features)

    return tracker[["Season", "GAME_ID", "Date", "HomeTeam", "AwayTeam", "PlayerNames", "Recommendation", "FeaturesUsed"]]
