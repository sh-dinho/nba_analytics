import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
import logging
import sqlite3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_PATH = "nba_analytics.db"
MODEL_PATH = "xgb_model.pkl"

# Create dummy data if nba_games is empty
con = sqlite3.connect(DB_PATH)
try:
    df = pd.read_sql("SELECT * FROM nba_games", con)
except:
    df = pd.DataFrame()
con.close()

if df.empty:
    logging.info("No NBA games found. Creating dummy data for model.")
    import numpy as np
    N = 100
    df = pd.DataFrame({
        "HOME_SCORE": np.random.randint(80, 130, N),
        "VISITOR_SCORE": np.random.randint(80, 130, N)
    })

df["TARGET"] = (df["HOME_SCORE"] > df["VISITOR_SCORE"]).astype(int)
df["SCORE_DIFF"] = df["HOME_SCORE"] - df["VISITOR_SCORE"]

X = df[["SCORE_DIFF"]]
y = df["TARGET"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
auc = roc_auc_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

logging.info(f"✔ Pretrained xgb_model.pkl created | Accuracy: {acc:.3f}, AUC: {auc:.3f}")
