import os
import sqlite3
import pandas as pd
import numpy as np
import yaml
import joblib
import logging
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------
# Config
# ---------------------------
# Root directory (one level above train/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.yaml")

CONFIG = yaml.safe_load(open(CONFIG_PATH))
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

CONFIG = yaml.safe_load(open(CONFIG_PATH))
DB_PATH = CONFIG["database"]["path"]
MODEL_DIR = CONFIG["model"]["models_dir"]
MODEL_FILE = os.path.join(MODEL_DIR, "xgb_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------
# Database Setup
# ---------------------------
def setup_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # NBA games table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS nba_games (
            GAME_ID TEXT PRIMARY KEY,
            GAME_DATE TEXT,
            TEAM_ABBREVIATION TEXT,
            PTS REAL,
            REB REAL,
            AST REAL,
            WL TEXT,
            MATCHUP TEXT
        )
    """)

    # Daily picks table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_picks (
            Timestamp TEXT,
            Team TEXT,
            Opponent TEXT,
            Probability REAL,
            Odds REAL,
            EV REAL,
            SuggestedStake REAL
        )
    """)

    # Bankroll tracker
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bankroll_tracker (
            Timestamp TEXT,
            StartingBankroll REAL,
            CurrentBankroll REAL,
            ROI REAL,
            Notes TEXT
        )
    """)

    # Model metrics
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            Timestamp TEXT,
            AUC REAL,
            Accuracy REAL
        )
    """)

    # Retrain history
    cur.execute("""
        CREATE TABLE IF NOT EXISTS retrain_history (
            Timestamp TEXT,
            ModelType TEXT,
            Status TEXT
        )
    """)

    con.commit()
    con.close()
    logging.info("✅ Database tables ensured.")

# ---------------------------
# Feature Engineering
# ---------------------------
def build_features(df):
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    # rolling averages
    for stat in ["PTS", "REB", "AST"]:
        df[f"{stat}_avg"] = df.groupby("TEAM_ABBREVIATION")[stat].transform(
            lambda x: x.rolling(CONFIG["features"]["rolling_games"], min_periods=1).mean()
        )
    # B2B / home
    df["last_game"] = df.groupby("TEAM_ABBREVIATION")["GAME_DATE"].shift(1)
    df["rest_days"] = (df["GAME_DATE"] - df["last_game"]).dt.days.fillna(2)
    df["b2b"] = (df["rest_days"] == 1).astype(int)
    df["home"] = df["MATCHUP"].str.contains("vs").astype(int)
    return df

def prepare_model_data(df):
    features = ["PTS_avg", "REB_avg", "AST_avg", "rest_days", "b2b", "home"]
    df = df.dropna(subset=features + ["WL"])
    X = df[features].values
    y = (df["WL"] == "W").astype(int).values
    return X, y

# ---------------------------
# Train XGBoost
# ---------------------------
def train_xgb(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_prob)
    acc = accuracy_score(y_test, y_pred)
    return model, auc, acc

# ---------------------------
# Save Model & Metrics
# ---------------------------
def save_model(model):
    joblib.dump(model, MODEL_FILE)
    logging.info(f"✔ Model saved to {MODEL_FILE}")

def save_metrics(auc, acc):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO model_metrics (Timestamp, AUC, Accuracy)
        VALUES (?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        float(auc),
        float(acc)
    ))
    con.commit()
    con.close()
    logging.info(f"✔ Metrics logged: AUC={auc:.3f}, Accuracy={acc:.3f}")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    setup_db()

    # Load NBA games data
    con = sqlite3.connect(DB_PATH)
    df_games = pd.read_sql("SELECT * FROM nba_games", con)
    con.close()

    if df_games.empty:
        logging.error("❌ No NBA game data found. Please fetch and store games first.")
        exit(1)

    df_feat = build_features(df_games)
    X, y = prepare_model_data(df_feat)

    model, auc, acc = train_xgb(X, y)
    save_model(model)
    save_metrics(auc, acc)
    logging.info("🚀 Training completed successfully!")
