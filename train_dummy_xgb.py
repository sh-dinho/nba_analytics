import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from datetime import datetime

# ------------------------
# Generate dummy NBA data
# ------------------------
np.random.seed(42)
n_games = 500

df = pd.DataFrame({
    "PTS_avg_team": np.random.randint(90, 130, n_games),
    "REB_avg_team": np.random.randint(30, 60, n_games),
    "AST_avg_team": np.random.randint(15, 35, n_games),
    "PTS_avg_opp": np.random.randint(90, 130, n_games),
    "REB_avg_opp": np.random.randint(30, 60, n_games),
    "AST_avg_opp": np.random.randint(15, 35, n_games),
    "rest_days": np.random.randint(0, 5, n_games),
    "b2b": np.random.randint(0, 2, n_games),
    "home": np.random.randint(0, 2, n_games),
    "target": np.random.randint(0, 2, n_games)  # win or lose
})

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# Train XGBoost model
# ------------------------
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ------------------------
# Evaluate
# ------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"Accuracy: {acc:.3f}, AUC: {auc:.3f}")

# ------------------------
# Save model
# ------------------------
joblib.dump(model, "xgb_model.pkl")
print(f"✔ Model saved as xgb_model.pkl at {datetime.now()}")
