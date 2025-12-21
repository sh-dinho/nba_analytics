"""
Model Training Pipeline
-----------------------
Loads long-format canonical data, builds ML features, trains a model,
and saves it to the model registry.

This version:
- Assumes all games from nba_api are completed ("final")
- Adds a synthetic `status` column
- Removes the broken filter `features_df["status"] == "final"`
"""

import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from src.features.builder import FeatureBuilder
from src.features.feature_store import FeatureStore
from src.config.paths import MODEL_REGISTRY_DIR
from loguru import logger


def train_model():
    # 1. Load latest data
    fs = FeatureStore()
    df_long, _ = fs.load_latest_snapshot()  # Or load from LONG_SNAPSHOT

    # 2. Build Features
    fb = FeatureBuilder(window=10)
    features_df = fb.build(df_long)

    # 3. Define features and target
    # We only train on rows where 'won' is known (final games)
    train_data = features_df.dropna(subset=["won"])

    # Use the feature list from the builder to ensure consistency
    feature_cols = fb.get_feature_names()
    X = train_data[feature_cols]
    y = train_data["won"]

    # 4. Train
    logger.info(f"Training RandomForest on {len(X)} rows...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)

    # 5. Save Model + Metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_REGISTRY_DIR / f"model_{timestamp}.joblib"

    # We save as a dict to keep the column names attached to the model
    model_data = {"model": model, "features": feature_cols, "timestamp": timestamp}
    joblib.dump(model_data, model_path)

    # Create a 'latest' symlink or pointer
    latest_path = MODEL_REGISTRY_DIR / "latest_model.joblib"
    joblib.dump(model_data, latest_path)

    logger.success(f"Model trained and saved to {latest_path}")


if __name__ == "__main__":
    train_model()
