# nba_analytics_core/predictor.py
import logging
import pandas as pd
from typing import Optional
from config import THRESHOLD
from nba_analytics_core.db_module import get_conn

def predict_todays_games(threshold: Optional[float] = None) -> pd.DataFrame:
    use_threshold = THRESHOLD if threshold is None else threshold
    logging.info(f"Predicting today's games with threshold={use_threshold}")
    # Example: Pull recent games and fabricate probabilities (replace with real model)
    with get_conn() as conn:
        df = pd.read_sql_query("SELECT g.game_id, g.home_team, g.away_team FROM games g ORDER BY date DESC LIMIT 10", conn)
    if df.empty:
        logging.warning("No games found for prediction")
        return pd.DataFrame(columns=["game_id", "home_team", "away_team", "predicted_prob", "predicted_winner"])
    df["predicted_prob"] = 0.62  # placeholder
    df["predicted_winner"] = df["home_team"].where(df["predicted_prob"] >= use_threshold, other=df["away_team"])
    logging.info(f"✔ Generated {len(df)} predictions")
    return df[["game_id", "home_team", "away_team", "predicted_prob", "predicted_winner"]]