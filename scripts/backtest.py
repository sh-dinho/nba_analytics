# Path: scripts/backtest.py
import os
import pandas as pd
import joblib
from nba_analytics_core.data import fetch_historical_games, build_team_stats, build_matchup_features
from scripts.train_model import MODEL_PATH, build_dataset

RESULTS_DIR = "results"
BACKTEST_FILE = os.path.join(RESULTS_DIR, "backtest.csv")

def run_backtest(seasons):
    """
    Run backtest on historical seasons using the trained classification model.
    Saves results/backtest.csv with actual vs predicted probabilities.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")

    model = joblib.load(MODEL_PATH)
    X, y = build_dataset(seasons)

    y_prob = model.predict_proba(X)[:, 1]

    # Build DataFrame with results
    df = pd.DataFrame({
        "season": [seasons[0]] * len(y),  # simple placeholder, can expand per season
        "home_win": y,
        "home_win_prob": y_prob
    })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(BACKTEST_FILE, index=False)
    print(f"📊 Backtest results saved to {BACKTEST_FILE}")

if __name__ == "__main__":
    seasons = ["2021-22", "2022-23", "2023-24"]
    run_backtest(seasons)