import os
import pandas as pd
import sqlite3
import yaml
import logging
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.db_init import init_db, ensure_table

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------------------
# Load config
# ------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

CONFIG = yaml.safe_load(open(CONFIG_PATH))

DB_PATH = CONFIG["database"]["path"]
MODEL_DIR = CONFIG["model"]["models_dir"]
MODEL_NAME = CONFIG["model"]["filename"]

os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------------------
# Feature Engineering
# ------------------------------
def build_features(df):
    logging.info("🔧 Building features...")

    # Simple engineered features
    df["PointDiff"] = df["HomePts"] - df["VisitorPts"]
    df["HomeWin"] = (df["HomePts"] > df["VisitorPts"]).astype(int)

    # Encode teams using numeric mapping
    teams = sorted(list(set(df["Home"]) | set(df["Visitor"])))
    team_map = {team: i for i, team in enumerate(teams)}

    df["HomeID"] = df["Home"].map(team_map)
    df["VisitorID"] = df["Visitor"].map(team_map)

    return df


# ------------------------------
# Train Model
# ------------------------------
def train_xgb_model():
    logging.info("📥 Loading game data...")

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("❌ Database not found. Run fetch_games.py first.")

    init_db(DB_PATH)
    ensure_table(DB_PATH)

    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM nba_games", con)
    con.close()

    if df.empty:
        raise ValueError("❌ No game data found. Run fetch_games.py first.")

    logging.info(f"📊 Loaded {len(df)} games")

    # Build features
    df = build_features(df)

    feature_cols = ["HomeID", "VisitorID"]
    target_col = "HomeWin"

    X = df[feature_cols]
    y = df[target_col]

    logging.info("🔀 Performing train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    logging.info("🚀 Training XGBoost model")

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist"
    )

    model.fit(X_train, y_train)

    # ------------------------------
    # Evaluate
    # ------------------------------
    predictions = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
    acc = accuracy_score(y_test, predictions)

    logging.info(f"📈 Accuracy: {acc:.4f}")

    # ------------------------------
    # Save Model with Versioning
    # ------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    versioned_name = f"xgb_model_{timestamp}.pkl"
    model_path = os.path.join(MODEL_DIR, versioned_name)

    model.save_model(model_path)

    logging.info(f"💾 Model saved: {model_path}")

    # Update latest model pointer
    latest_path = os.path.join(MODEL_DIR, MODEL_NAME)
    if os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(model_path, latest_path) if hasattr(os, "symlink") else None

    # ------------------------------
    # Store metrics in DB
    # ------------------------------
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO model_history (Timestamp, Version, Accuracy)
        VALUES (?, ?, ?)
    """, (timestamp, versioned_name, float(acc)))

    conn.commit()
    conn.close()

    logging.info("📚 Training logged in database")

    return {
        "model_version": versioned_name,
        "accuracy": acc,
        "path": model_path
    }


# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    result = train_xgb_model()
    logging.info("🎉 Training complete.")
    logging.info(result)
