from xgboost import XGBClassifier
import joblib
import logging
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from app.database import fetch_games

MODEL_PATH = 'models/xgb_model.pkl'

def train_xgb_model():
    """Train the XGBoost model with the NBA data."""
    df = fetch_games()
    if df.empty:
        logging.error("No NBA game data found to train the model.")
        return None

    FEATURES = ['home_score', 'away_score', 'home_team_win_pct', 'away_team_win_pct']
    TARGET = 'home_win'

    # Validate features
    for f in FEATURES:
        if f not in df.columns:
            logging.error(f"Missing feature: {f}")
            return None

    X = df[FEATURES].fillna(0)
    y = df[TARGET]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    logging.info(f"Validation accuracy: {acc:.2f}")

    # Ensure directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Save model
    joblib.dump(model, MODEL_PATH)
    logging.info(f"Model trained and saved to {MODEL_PATH}")
    return model