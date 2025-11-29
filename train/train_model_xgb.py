import sqlite3
import pandas as pd
import joblib
import os
import yaml
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]
MODEL_DIR = CONFIG["model"]["models_dir"]
CURRENT_MODEL = CONFIG["model"]["filename"]

os.makedirs(MODEL_DIR, exist_ok=True)

def fetch_features_labels():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM nba_games", con)
    con.close()
    if df.empty:
        return None, None

    # Example features
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])
    df['PTS_avg'] = df.groupby('TEAM_ABBREVIATION')['PTS'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['REB_avg'] = df.groupby('TEAM_ABBREVIATION')['REB'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['AST_avg'] = df.groupby('TEAM_ABBREVIATION')['AST'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df = df.dropna(subset=['PTS_avg', 'REB_avg', 'AST_avg'])

    X = df[['PTS_avg','REB_avg','AST_avg']]
    y = (df['WL'] == 'W').astype(int)
    return X, y

def train_model():
    X, y = fetch_features_labels()
    if X is None or X.empty:
        print("❌ No data to train")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Save versioned model
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    version_file = os.path.join(MODEL_DIR, f"xgb_model_{timestamp}.joblib")
    joblib.dump(model, version_file)

    # Update current model symlink
    current_path = os.path.join("models", CURRENT_MODEL)
    if os.path.exists(current_path):
        os.remove(current_path)
    os.symlink(version_file, current_path)

    # Log metrics
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO model_metrics (Timestamp, Accuracy, AUC) VALUES (?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M"), float(acc), float(auc)))
    con.execute("INSERT INTO retrain_history (Timestamp, ModelVersion, Status) VALUES (?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M"), version_file, "success"))
    con.commit()
    con.close()
    print(f"✔ Model trained: Acc={acc:.3f}, AUC={auc:.3f}")

if __name__ == "__main__":
    train_model()

