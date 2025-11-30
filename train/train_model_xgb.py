import sqlite3
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
import os
import yaml
import logging

logging.basicConfig(level=logging.INFO)

CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]
MODEL_DIR = CONFIG["model"]["models_dir"]
MODEL_PATH = os.path.join(MODEL_DIR, CONFIG["model"]["filename"])

os.makedirs(MODEL_DIR, exist_ok=True)

def train_xgb_model():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM nba_games", con)
    con.close()

    if df.empty:
        raise ValueError("No NBA game data found. Fetch games first.")

    # Simplified features for demonstration
    df['home_win'] = (df['home_score'] > df['visitor_score']).astype(int)
    df['score_diff'] = df['home_score'] - df['visitor_score']

    X = df[['home_score', 'visitor_score', 'score_diff']]
    y = df['home_win']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    logging.info(f"✔ Model trained | Accuracy: {acc:.3f} | AUC: {auc:.3f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"💾 Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_xgb_model()
