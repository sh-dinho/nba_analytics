# scripts/simulate_ai_strategy.py
import logging
import pandas as pd
import numpy as np

# Assuming this function retrieves a historical dataset including predictions and actual results ('winner')
# from app.predictor import predict_todays_games # Renamed for conceptual clarity in a backtest scenario
from app.predict_pipeline import calculate_expected_value 
from scripts.simulate_bankroll import simulate_bankroll


def predict_historical_games() -> pd.DataFrame:
    """
    Placeholder for a function that fetches historical game data that has 
    already been processed by the ML models and includes columns like:
    'game_id', 'home_team', 'away_team', 'pred_home_win_prob', 'home_decimal_odds', 'winner'
    
    NOTE: In a real environment, this would call a pipeline to load and predict 
    on historical data, ensuring 'winner' is present.
    """
    logging.warning("Using placeholder historical data for simulation.")
    # Create mock historical data for demonstration
    data = {
        'game_id': [1, 2, 3, 4, 5],
        'home_team': ['Team A', 'Team B', 'Team C', 'Team D', 'Team E'],
        'away_team': ['Team X', 'Team Y', 'Team Z', 'Team W', 'Team V'],
        'pred_home_win_prob': [0.65, 0.52, 0.61, 0.58, 0.70],
        # Decimal odds for +150, -110, +100, -120, -110
        'home_decimal_odds': [2.50, 1.91, 2.00, 1.83, 1.91], 
        'winner': ['Team A', 'Team Y', 'Team Z', 'Team D', 'Team V'] # Team Y means Away Win (0) for Team B
    }
    df = pd.DataFrame(data)
    
    # Calculate initial EV based on mock data
    df["home_ev"] = df.apply(
        lambda r: calculate_expected_value(r["pred_home_win_prob"], r["home_decimal_odds"]),
        axis=1
    )
    return df


def simulate_ai_strategy(initial_bankroll: int = 1000, strategy: str = "kelly", threshold: float = 0.6) -> pd.DataFrame:
    """
    Runs a historical backtest simulation of the AI strategy.
    Bets are placed only when:
    1. Predicted Probability >= threshold (e.g., 60%)
    2. Expected Value (EV) > 0
    """
    # Using historical data source for backtesting
    df_history = predict_historical_games() 
    
    if df_history.empty:
        logging.info("No historical games available for AI simulation.")
        return pd.DataFrame()

    bets = []
    
    for _, row in df_history.iterrows():
        prob = row["pred_home_win_prob"]
        odds = row["home_decimal_odds"]
        
        # 1. Check Probability Threshold
        if prob >= threshold:
            
            # 2. Check for Positive Expected Value (EV > 0)
            # We must recalculate EV or assume 'home_ev' is present from the historical pipeline
            ev = row["home_ev"]
            
            if ev > 0:
                # Bet qualifies. Build the bet structure aligned with the pipeline's output.
                # Note: We use the actual result ('winner') for the simulation.
                is_win = 1 if row["winner"] == row["home_team"] else 0
                
                bets.append({
                    "game_id": row["game_id"],
                    "team": row["home_team"],
                    "decimal_odds": odds,
                    "prob": prob,
                    "ev": ev,
                    "result": is_win # This is required for backtesting/simulation
                })

    if not bets:
        logging.info("No qualifying value bets from AI predictions.")
        return pd.DataFrame()

    df_bets = pd.DataFrame(bets)
    
    # Run the simulation using the historical results
    # The simulate_bankroll function will use 'prob', 'decimal_odds', and 'ev' 
    # to determine the dynamic stake based on the 'strategy'.
    sim_results = simulate_bankroll(df_bets, initial_bankroll=initial_bankroll, strategy=strategy)
    
    logging.info(f"Simulation complete. Strategy: {strategy}, Total Bets: {len(df_bets)}")
    return sim_results