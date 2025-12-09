# ============================================================
# Path: src/features/generate_features.py
# Purpose: Generate features for NBA games with guaranteed GAME_ID and TEAM_ID
# Version: 1.3 (final patch)
# ============================================================

import pandas as pd
import logging

def generate_features_for_games(game_data_list):
    """
    Generate features for a list of games.
    Ensures GAME_ID and TEAM_ID columns are always present.
    Filters out empty DataFrames to avoid FutureWarning.
    """
    features = []

    for i, game in enumerate(game_data_list):
        df = pd.DataFrame([game])

        # Guarantee GAME_ID
        if "GAME_ID" not in df.columns:
            if "game_id" in game:
                df["GAME_ID"] = game["game_id"]
            else:
                logging.warning("Missing GAME_ID in raw data, assigning placeholder")
                df["GAME_ID"] = f"unknown_game_{i}"

        # Guarantee TEAM_ID
        if "TEAM_ID" not in df.columns:
            if "team_id" in game:
                df["TEAM_ID"] = game["team_id"]
            else:
                logging.warning("Missing TEAM_ID in raw data, assigning placeholder")
                df["TEAM_ID"] = -1

        features.append(df)

    # Filter out empty DataFrames before concatenation
    valid_features = [df for df in features if not df.empty]
    if not valid_features:
        logging.warning("No valid features generated")
        return pd.DataFrame()

    return pd.concat(valid_features, ignore_index=True)
