import sqlite3
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import logging
import joblib
import os

logging.basicConfig(level=logging.INFO)

DB_PATH = "db/nba_analytics.db"
MODEL_PATH = "models/xgb_model.pkl"
os.makedirs("models", exist_ok=True)

def train_xgb_model():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM nba_games", con)
    con.close()

    if df.empty:
        logging.error("No NBA game data found. Fetch games first.")
        return

    # Feature engineering
    df["HomeWin"] = (df["Home_PTS"] > df["Visitor_PTS"]).astype(int)
    df["PTS_Diff"] = df["Home_PTS"] - df["Visitor_PTS"]

    X = df[["PTS_Diff"]]
    y = df["HomeWin"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    logging.info(f"Model trained. Accuracy: {acc:.3f}, AUC: {auc:.3f}")

    joblib.dump(model, MODEL_PATH)
    logging.info(f"✔ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_xgb_model()
