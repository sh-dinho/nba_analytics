"""
Runner script for training the NBA Analytics v3 model.
Uses the functional Feature Store API to load data and save trained models.
"""

from loguru import logger
from src.features.feature_store import load_latest_snapshot
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
import json

# Model storage path
MODEL_PATH = Path("models/registry")
MODEL_PATH.mkdir(parents=True, exist_ok=True)


def run_training():
    """
    Function that trains the NBA model using features from the Feature Store.
    Saves the trained model and metadata in the model registry.

    Returns:
        dict: Metadata of the trained model
    """
    logger.info("🚀 Starting NBA v3 model training...")

    # Step 1: Load features from the Feature Store
    try:
        df = load_latest_snapshot()
    except FileNotFoundError:
        logger.error(
            "Feature Store data not found. Please ensure the pipeline has ingested data correctly."
        )
        return

    # Define the features and target
    features = ["home_minus_away"]  # Example feature: Difference in home and away score
    target = "homeWin"  # Target: Predicting if the home team wins

    # Step 2: Prepare the data (X = features, y = target)
    X = df[features]
    y = df[target]

    logger.info(f"Training model with {len(X)} data points...")

    # Step 3: Initialize and train the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Step 4: Save the model and metadata
    version = len(list(MODEL_PATH.glob("model_*.joblib"))) + 1
    model_file = MODEL_PATH / f"model_{version}.joblib"
    meta_file = MODEL_PATH / f"model_{version}.meta.json"

    # Save model to disk
    joblib.dump(model, model_file)

    # Save metadata (version and features used for training)
    meta_data = {
        "version": version,
        "features": features,
        "model_type": "RandomForestClassifier",
    }

    with open(meta_file, "w") as f:
        json.dump(meta_data, f, indent=4)

    logger.info(f"Training complete. Model version → {version}")
    logger.info(f"Model saved to {model_file}")
    logger.info(f"Metadata saved to {meta_file}")

    return {"version": version, "model_file": model_file, "meta_file": meta_file}


# Run the training function
if __name__ == "__main__":
    run_training()
