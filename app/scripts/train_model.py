# scripts/train_model.py
import sqlite3
import logging
import joblib
import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

logging.basicConfig(level=logging.INFO)

DB_PATH = "data/nba_analytics.db"

def train_xgb_model():
    # Fetch data from the database
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM nba_games", con)
    con.close()

    if df.empty:
        logging.error("❌ No NBA game data found.")
        return None

    # Feature engineering: add more features here
    features = ["home_team_score", "away_team_score"]
    target = "home_win"

    X = df[features].fillna(0)
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost model
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    model_path = "models/xgb_model.pkl"
    joblib.dump(model, model_path)

    logging.info(f"✔ Model trained and saved to {model_path}")
    logging.info(f"✔ Accuracy: {acc:.3f}, AUC: {auc:.3f}")

    return acc, auc

if __name__ == "__main__":
    train_xgb_model()