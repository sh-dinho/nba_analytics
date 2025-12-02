import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
import joblib
from scripts.utils import setup_logger

logger = setup_logger("train_model")
DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    features_file = os.path.join(DATA_DIR, "training_features.csv")
    model_file = os.path.join(MODELS_DIR, "game_predictor.pkl")

    if not os.path.exists(features_file):
        logger.error(f"{features_file} not found. Build features first.")
        raise FileNotFoundError(features_file)

    logger.info("Training model...")

    df = pd.read_csv(features_file)

    # Features and target
    X = df[[
        "home_pts_avg", "away_pts_avg",
        "home_ast_avg", "away_ast_avg",
        "home_reb_avg", "away_reb_avg"
    ]]
    y = df["home_win"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_prob),
        "brier": brier_score_loss(y_test, y_prob),
        "auc": roc_auc_score(y_test, y_prob)
    }

    joblib.dump(clf, model_file)
    logger.info(f"Model saved to {model_file}")
    logger.info("Training metrics:")
    for k,v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    return metrics

if __name__ == "__main__":
    main()
