import pandas as pd
import sqlite3
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
import logging
import yaml
from datetime import datetime

logging.basicConfig(level=logging.INFO)

CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]
MODEL_PATH = "xgb_model.pkl"

def train_model():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM nba_games", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()

    if df.empty:
        logging.warning("No NBA games found. Using dummy data.")
        import numpy as np
        df = pd.DataFrame({"HOME_SCORE": np.random.randint(80, 130, 100),
                           "VISITOR_SCORE": np.random.randint(80, 130, 100)})
    
    df["TARGET"] = (df["HOME_SCORE"] > df["VISITOR_SCORE"]).astype(int)
    df["SCORE_DIFF"] = df["HOME_SCORE"] - df["VISITOR_SCORE"]

    X = df[["SCORE_DIFF"]]
    y = df["TARGET"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"✔ Model trained | Accuracy: {acc:.3f} | AUC: {auc:.3f}")

if __name__ == "__main__":
    train_model()
