import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from scripts.utils import setup_logger, safe_mkdir

logger = setup_logger("train_model")

def main(features_file="data/training_features.csv", model_file="models/game_predictor.pkl"):
    df = pd.read_csv(features_file)
    X = df[["points_norm", "assists_norm", "rebounds_norm"]]
    y = df["target_home_win"]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    safe_mkdir("models")
    joblib.dump(clf, model_file)
    logger.info(f"Model saved to {model_file}")

    # Example metrics
    metrics = {
        "accuracy": clf.score(X, y),
        "n_samples": len(df)
    }
    return metrics
