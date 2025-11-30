import logging
from pathlib import Path
import sqlite3
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Paths
BASE_DIR = Path(__file__).parent.parent.resolve()
CONFIG_PATH = BASE_DIR / "config.yaml"
DB_PATH = BASE_DIR / "nba_analytics.db"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "xgb_model.pkl"

# Load config
import yaml
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
CONFIG = yaml.safe_load(open(CONFIG_PATH))

# Load game data
con = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM nba_games", con)
con.close()

if df.empty:
    raise ValueError("No NBA game data found. Please fetch and store games first.")

# Example: use basic features
FEATURES = ["HomePTS", "AwayPTS", "HomeFG%", "AwayFG%"]
TARGET = "HomeWin"  # 1 if home team won, 0 otherwise

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
auc = roc_auc_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

joblib.dump(model, MODEL_PATH)
logging.info(f"✔ Model trained and saved to {MODEL_PATH}")
logging.info(f"✔ Accuracy: {acc:.3f}, AUC: {auc:.3f}")
