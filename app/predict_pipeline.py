# app/predict_pipeline.py
import logging
import pandas as pd
from nba_analytics_core.fetch_games import get_todays_games
from nba_analytics_core.data import engineer_features
from nba_analytics_core.odds import fetch_game_odds
from app.predictor import load_models # your existing load_models
from nba_analytics_core.utils import send_telegram_message

def calculate_expected_value(prob: float, odds: float) -> float:
    """
    Calculates the Expected Value (EV) for a bet.
    EV = (P_pred * (Odds - 1)) - ((1 - P_pred) * 1)
    """
    # Check if odds are valid (must be greater than 1.0)
    if odds <= 1.0:
        return -1.0
        
    # P_win * Net_Profit - P_loss * Loss
    return (prob * (odds - 1)) - ((1 - prob) * 1)


def generate_predictions(threshold: float = 0.6) -> pd.DataFrame:
    games = get_todays_games()
    if games.empty:
        logging.info("No games found for today.")
        return pd.DataFrame()

    df_features = engineer_features(games)
    clf, reg = load_models()
    if clf is None or reg is None:
        logging.error("Models not available.")
        return pd.DataFrame()

    drop_cols = [c for c in ["home_win", "total_points"] if c in df_features.columns]
    X = df_features.drop(columns=drop_cols, errors="ignore")

    try:
        df_features["pred_home_win_prob"] = clf.predict_proba(X)[:, 1]
        df_features["pred_total_points"] = reg.predict(X)
        # Strong pick now means meeting the probability threshold
        df_features["strong_pick"] = (df_features["pred_home_win_prob"] >= threshold).astype(int)
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return pd.DataFrame()

    # Merge back to games
    preds = games.merge(
        df_features[["game_id", "pred_home_win_prob", "pred_total_points", "strong_pick"]],
        on="game_id"
    )

    # Attach odds
    odds = fetch_game_odds(games)
    preds = preds.merge(odds, on=["game_id", "home_team", "away_team"], how="left")
    
    # --- IMPROVEMENT: Calculate EV ---
    preds["home_ev"] = preds.apply(
        lambda r: calculate_expected_value(r["pred_home_win_prob"], r["home_decimal_odds"]),
        axis=1
    )

    # Notify if strong picks found
    strong = preds[preds["strong_pick"] == 1]
    if not strong.empty:
        try:
            send_telegram_message(f"Strong picks: {len(strong)} found today ✅")
        except Exception:
            pass

    return preds

def build_bets_from_predictions(preds: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    """
    Identifies qualifying bets based on the probability threshold AND positive Expected Value (EV).
    """
    if preds.empty:
        return pd.DataFrame()
    
    tickets = []
    for _, r in preds.iterrows():
        # Check 1: Must meet the probability threshold
        if r["pred_home_win_prob"] >= threshold:
            
            # Check 2: Must have a positive Expected Value (EV > 0)
            # The odds offered must be higher than the implied odds of the model's prediction
            if r["home_ev"] > 0:
                tickets.append({
                    "game_id": r["game_id"],
                    "team": r["home_team"],
                    "decimal_odds": r["home_decimal_odds"],
                    "prob": r["pred_home_win_prob"],
                    "ev": r["home_ev"], # New column
                    "result": None if pd.isna(r.get("winner")) else int(r["winner"] == r["home_team"])
                })
    return pd.DataFrame(tickets)