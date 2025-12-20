"""
Runner script for batch prediction.
"""

from loguru import logger
from pathlib import Path
import pandas as pd

from src.model.train import load_model, load_metadata
from src.model.predict import predict_games
from src.features.feature_store import FeatureStore, FEATURE_PATH


def ensure_parquet_from_csv():
    """
    Ensure that a Parquet snapshot exists from the canonical CSV ingestion.
    If Parquet already exists, do nothing. Otherwise, convert CSV → Parquet.
    """
    csv_path = Path("data/raw/schedule.csv")
    if not csv_path.exists():
        logger.error("Canonical CSV ingestion not found at data/raw/schedule.csv")
        return

    # Check if any Parquet snapshot exists
    if FEATURE_PATH.exists() and any(FEATURE_PATH.glob("features_*.parquet")):
        logger.info("Parquet feature snapshot already exists. Skipping CSV conversion.")
        return

    # Convert CSV → Parquet
    df_csv = pd.read_csv(csv_path)
    FEATURE_PATH.mkdir(parents=True, exist_ok=True)
    parquet_path = (
        FEATURE_PATH
        / f"features_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    )
    df_csv.to_parquet(parquet_path, index=False)
    logger.info(f"Converted schedule.csv → Parquet feature snapshot: {parquet_path}")


def run_prediction(model=None, metadata=None, df_features=None):
    """
    Run batch predictions for scheduled NBA games.

    Args:
        model: Optional pre-loaded model (otherwise loads latest from registry)
        metadata: Optional pre-loaded model metadata
        df_features: Optional features DataFrame (otherwise loads from Feature Store)

    Returns:
        Tuple[pd.DataFrame, str]: predicted DataFrame, path to predictions parquet
    """
    logger.info("Running NBA v3 prediction...")

    # ---------------------------
    # Ensure Parquet snapshot exists
    # ---------------------------
    ensure_parquet_from_csv()

    # ---------------------------
    # Load features if not provided
    # ---------------------------
    if df_features is None:
        store = FeatureStore()
        df_features = store.load_latest_snapshot()

        # Only predict scheduled/future games
        if "status" in df_features.columns:
            df_features = df_features[df_features["status"] == "scheduled"]

        if df_features.empty:
            logger.info("No scheduled games found for prediction.")
            return None, None

    # ---------------------------
    # Load model if not provided
    # ---------------------------
    if model is None:
        model = load_model()
    if metadata is None:
        metadata = load_metadata()

    # ---------------------------
    # Predict
    # ---------------------------
    df_preds, path = predict_games(
        model=model,
        metadata=metadata,
        df_features=df_features,
        output_dir="data/predictions",
    )

    logger.info(f"Prediction complete → {path}")
    return df_preds, path


# ---------------------------
# CLI entrypoint
# ---------------------------
if __name__ == "__main__":
    run_prediction()
