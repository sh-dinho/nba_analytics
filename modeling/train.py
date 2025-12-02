# modeling/train.py
import pandas as pd
import pickle
from core.logging import setup_logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from core.paths import MODELS_DIR
import os

logger = setup_logger("train_model")

def train_model(features_df, target="home_win"):
    X = features_df.drop(columns=[target], errors="ignore")
    y = features_df[target] if target in features_df else pd.Series([1]*len(X))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "log_loss": float(log_loss(y_test, y_prob)),
        "brier": float(brier_score_loss(y_test, y_prob)),
        "auc": float(roc_auc_score(y_test, y_prob))
    }
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_file = os.path.join(MODELS_DIR, "game_predictor.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_file}")
    return model_file, metrics
