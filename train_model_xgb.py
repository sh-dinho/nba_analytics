import sqlite3
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import pickle
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_PATH = "nba_analytics.db"
MODEL_PATH = "xgb_model.pkl"

# -----------------------------
# 1️⃣ Load games from database
# -----------------------------
def load_games():
    try:
        con = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM nba_games", con)
        con.close()
        if df.empty:
            raise ValueError("No NBA game data found. Please fetch and store games first.")
        logging.info(f"✅ Loaded {len(df)} games from database")
        return df
    except Exception as e:
        logging.error(f"❌ Failed to load games: {e}")
        return pd.DataFrame()

# -----------------------------
# 2️⃣ Feature Engineering
# -----------------------------
def prepare_features(df):
    df = df.copy()
    # Basic feature: home win (target)
    df["HOME_WIN"] = (df["HOME_SCORE"] > df["VISITOR_SCORE"]).astype(int)
    
    # Feature: home team score diff previous game
    df["SCORE_DIFF"] = df["HOME_SCORE"] - df["VISITOR_SCORE"]
    
    # Encode teams as categorical integers
    teams = pd.concat([df["HOME_TEAM"], df["VISITOR_TEAM"]]).unique()
    team_map = {team: idx for idx, team in enumerate(teams)}
    df["HOME_TEAM_ID_ENC"] = df["HOME_TEAM"].map(team_map)
    df["VISITOR_TEAM_ID_ENC"] = df["VISITOR_TEAM"].map(team_map)
    
    # Features and target
    X = df[["HOME_TEAM_ID_ENC", "VISITOR_TEAM_ID_ENC", "SCORE_DIFF"]]
    y = df["HOME_WIN"]
    return X, y

# -----------------------------
# 3️⃣ Train XGBoost
# -----------------------------
def train_xgb(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Metrics
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    logging.info(f"✔ Training complete | AUC: {auc:.3f} | Accuracy: {acc:.3f}")
    return model, auc, acc

# -----------------------------
# 4️⃣ Save model
# -----------------------------
def save_model(model):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"✔ Model saved to {MODEL_PATH}")

# -----------------------------
# 5️⃣ Store metrics
# -----------------------------
def store_metrics(auc, acc):
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                Timestamp TEXT,
                ModelType TEXT,
                AUC REAL,
                Accuracy REAL
            )
        """)
        con.execute("""
            INSERT INTO model_metrics (Timestamp, ModelType, AUC, Accuracy)
            VALUES (?, ?, ?, ?)
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "xgboost", auc, acc))
        con.commit()
        con.close()
        logging.info("✔ Model metrics stored in database")
    except Exception as e:
        logging.error(f"❌ Failed to store metrics: {e}")

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    df_games = load_games()
    if not df_games.empty:
        X, y = prepare_features(df_games)
        model, auc, acc = train_xgb(X, y)
        save_model(model)
        store_metrics(auc, acc)
