import pandas as pd
import xgboost as xgb
import sqlite3
import joblib
import os
import logging
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

logging.basicConfig(level=logging.INFO)
CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]
MODEL_PATH = os.path.join(CONFIG["model"]["models_dir"], CONFIG["model"]["filename"])
os.makedirs(CONFIG["model"]["models_dir"], exist_ok=True)

# Load data
con = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM nba_games", con)
con.close()

if df.empty:
    raise ValueError("No NBA game data found. Fetch games first.")

df = df.dropna()
X = df[["HomePTS", "AwayPTS", "Season"]]
y = (df["HomePTS"] > df["AwayPTS"]).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, y_prob)
acc = accuracy_score(y_test, y_pred)
logging.info(f"Trained XGB | AUC={auc:.3f}, Accuracy={acc:.3f}")

joblib.dump(model, MODEL_PATH)
logging.info(f"Model saved to {MODEL_PATH}")
