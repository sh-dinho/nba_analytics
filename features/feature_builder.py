# features/feature_builder.py
import pandas as pd
from core.logging import setup_logger
from data_ingest.fetch_player_stats import fetch_player_stats

logger = setup_logger("feature_builder")

def build_features(season="2024-25"):
    stats_df = fetch_player_stats(season)
    
    # Placeholder: simple features
    features_df = stats_df.groupby("team").agg({
        "pts":"mean",
        "ast":"mean",
        "reb":"mean"
    }).reset_index()
    features_df.columns = ["team","avg_pts","avg_ast","avg_reb"]
    logger.info(f"Built features for {len(features_df)} teams")
    return features_df
