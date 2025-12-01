# app/predict_pipeline.py
import logging
import pandas as pd
from core.fetch_games import get_todays_games
from core.data import engineer_features
from core.odds import fetch_game_odds
from app.predictor import load_models  # your existing load_models
from core.utils import send_telegram_message

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

    # Notify if strong picks found
    strong = preds[preds["strong_pick"] == 1]
    if not strong.empty:
        try:
            send_telegram_message(f"Strong picks: {len(strong)} found today ✅")
        except Exception:
            pass

    return preds

def build_bets_from_predictions(preds: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()
    tickets = []
    for _, r in preds.iterrows():
        if r["pred_home_win_prob"] >= threshold:
            tickets.append({
                "game_id": r["game_id"],
                "team": r["home_team"],
                "decimal_odds": r["home_decimal_odds"],
                "prob": r["pred_home_win_prob"],
                "result": None if pd.isna(r.get("winner")) else int(r["winner"] == r["home_team"])
            })
    return pd.DataFrame(tickets)