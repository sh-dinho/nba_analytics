# pipelines/daily_pipeline.py
from data_ingest.fetch_player_stats import fetch_player_stats
from data_ingest.fetch_odds import fetch_odds
from features.feature_builder import build_features
from modeling.train import train_model
from modeling.predict import predict
from betting.ev_calculator import calculate_ev
from betting.picks import generate_picks
from betting.bankroll import simulate_bankroll
from core.logging import setup_logger
from core.paths import ensure_dirs

logger = setup_logger("daily_pipeline")

def run_daily_pipeline(threshold=0.6, strategy="kelly", max_fraction=0.05):
    ensure_dirs()
    
    # 1) Fetch data
    fetch_player_stats()
    fetch_odds()
    
    # 2) Features
    features_df = build_features()
    
    # 3) Train model
    model_file, metrics = train_model(features_df)
    
    # 4) Predict
    preds_df = predict(features_df, model_file)
    
    # 5) Calculate EV
    preds_df = calculate_ev(preds_df)
    
    # 6) Generate picks
    picks_df = generate_picks(preds_df, threshold)
    
    # 7) Simulate bankroll
    trajectory, bankroll_metrics = simulate_bankroll(picks_df, strategy=strategy, max_fraction=max_fraction)
    
    picks_df["bankroll"] = trajectory
    return picks_df, bankroll_metrics
