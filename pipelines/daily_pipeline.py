# File: pipelines/daily_pipeline.py

from data_ingest.fetch_player_stats import fetch_player_stats
from data_ingest.fetch_odds import fetch_odds
from features.feature_builder import build_features
# ... (rest of imports)

logger = setup_logger("daily_pipeline")

# FIX 1: Added use_synthetic argument to function signature
def run_daily_pipeline(threshold=0.6, strategy="kelly", max_fraction=0.05, use_synthetic=False):
    ensure_dirs()
    
    # 1) Fetch data
    # FIX 2: Passed the use_synthetic flag to the fetcher
    fetch_player_stats(use_synthetic=use_synthetic)
    fetch_odds()
    
    # 2) Features
    features_df = build_features()
    
    # ... (rest of pipeline logic)
    
    picks_df["bankroll"] = trajectory
    return picks_df, bankroll_metrics