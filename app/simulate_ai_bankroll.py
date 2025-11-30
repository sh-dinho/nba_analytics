import logging
import pandas as pd

from app.predictor import predict_todays_games
from scripts.simulate_bankroll import simulate_bankroll


def simulate_ai_strategy(initial_bankroll=1000, strategy="kelly"):
    df_pred = predict_todays_games()
    if df_pred.empty:
        logging.info("No games available for AI simulation.")
        return pd.DataFrame()

    bets = []
    for _, row in df_pred.iterrows():
        if row["pred_home_win_prob"] > 0.55:
            bets.append({
                "game_id": row["game_id"],
                "team": row["home_team"],
                "odds": -110,  # placeholder odds
                "stake": 1.0,
                "result": 1 if row["winner"] == row["home_team"] else 0
            })

    if not bets:
        logging.info("No qualifying bets from AI predictions.")
        return pd.DataFrame()

    df_bets = pd.DataFrame(bets)
    return simulate_bankroll(df_bets, initial_bankroll=initial_bankroll, strategy=strategy)