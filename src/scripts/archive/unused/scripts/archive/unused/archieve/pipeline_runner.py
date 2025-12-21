"""
Runner script for full pipeline including ingestion, feature store, training, and prediction.
Used by: make pipeline
"""

from loguru import logger
from src.features.feature_store import FeatureStore
from src.model.train import train_model
from src.model.predict import predict_games
from src.model.registry import ModelRegistry


def run_full_pipeline():
    logger.info("🚀 Starting NBA Analytics v3 full pipeline")

    # ---------------------------
    # Step 1: Data Ingestion
    # ---------------------------
    logger.info("Step 1/5 — Ingestion")
    store = FeatureStore()

    store.ensure_parquet()

    # ---------------------------
    # Step 2: Feature Store
    # ---------------------------
    logger.info("Step 2/5 — Feature Store")
    df_features = store.load_latest_snapshot()

    # Check if there are any scheduled games to predict
    if "status" in df_features.columns:
        df_features = df_features[df_features["status"] == "scheduled"]

    if df_features.empty:
        logger.info("No scheduled games found for prediction.")
        return

    # ---------------------------
    # Step 3: Model Training
    # ---------------------------
    logger.info("Step 3/5 — Training the Model")
    registry = ModelRegistry()  # Create a ModelRegistry instance

    # Train the model using the features from the feature store (pass df_features and registry)
    metadata = train_model(
        feature_df=df_features,  # Pass the DataFrame instead of the feature store object
        registry=registry,  # Pass the registry object
        target_col="homeWin",  # Example: predicting if the home team wins
    )

    logger.info(f"Training complete. Model version → {metadata['version']}")

    # ---------------------------
    # Step 4: Make Predictions
    # ---------------------------
    logger.info("Step 4/5 — Making Predictions")
    model, metadata, version = (
        registry.load_latest()
    )  # Load the latest model and metadata

    if model is None:
        logger.error("No model found in registry!")
        return

    predictions, predictions_path = predict_games(
        model=model,
        metadata=metadata,
        df_features=df_features,
        output_dir="data/predictions",
    )

    logger.info(f"Prediction complete → {predictions_path}")

    # ---------------------------
    # Step 5: Finalization
    # ---------------------------
    logger.info("Step 5/5 — Finalization")
    logger.info("NBA v3 pipeline complete!")


# ---------------------------
# CLI entrypoint
# ---------------------------
if __name__ == "__main__":
    run_full_pipeline()
