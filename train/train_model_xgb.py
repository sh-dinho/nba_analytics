import os
import pickle
import logging
import sqlite3
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

CONFIG = yaml.safe_load(open(CONFIG_PATH))
DB_PATH = CONFIG["database"]["path"]
MODEL_DIR = CONFIG["model"]["models_dir"]
MODEL_FILE = os.path.join(MODEL_DIR, CONFIG["model"]["filename"])

os.makedirs(MODEL_DIR, exist_ok=True)

def train_xgb_model():
    # Load NBA games data
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM nba_games", con)
    con.close()

    if df.empty:
        raise ValueError("No NBA game data found. Please fetch and store games first.")

    # Features & target
    features = ["home_team_score", "away_team_score", "home_team_rest", "away_team_rest"]  # example
    target = "home_win"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    logging.info(f"✔ Model trained. Accuracy={acc:.3f}, AUC={auc:.3f}")

    # Save model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"✔ Model saved at {MODEL_FILE}")

    return acc, auc

if __name__ == "__main__":
    train_xgb_model()
