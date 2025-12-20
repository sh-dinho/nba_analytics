"""
Batch prediction over scheduled games using the merged ingestion snapshot.

It predicts for all games with status = 'scheduled' in:
    data/ingestion/ingestion_snapshot.parquet

Output:
    data/predictions/predictions_batch.parquet
"""

from pathlib import Path
import pandas as pd

from src.features.feature_builder import PreGameFeatureBuilder
from src.model.train import load_model, load_metadata


def predict_batch() -> pd.DataFrame:
    snapshot_path = Path("data/ingestion/ingestion_snapshot.parquet")
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Ingestion snapshot not found: {snapshot_path}")

    df = pd.read_parquet(snapshot_path)

    required_cols = {"game_id", "date", "home_team", "away_team", "status"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Ingestion snapshot missing columns: {missing}")

    # Only predict for scheduled (future) games
    df_future = df[df["status"] == "scheduled"].copy()
    if df_future.empty:
        print("No scheduled games found for batch prediction.")
        return df_future

    # Build pre-game features
    builder = PreGameFeatureBuilder(snapshot_path=snapshot_path)
    df_features = builder.build_for_games(df_future)

    model = load_model()
    metadata = load_metadata()
    feature_cols = metadata["features_used"]

    # Ensure all required features exist
    missing_features = set(feature_cols) - set(df_features.columns)
    if missing_features:
        raise ValueError(f"Feature builder did not produce columns: {missing_features}")

    df_features["probability"] = model.predict_proba(df_features[feature_cols])[:, 1]

    out = Path("data/predictions/predictions_batch.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(out, index=False)

    print(f"Saved batch predictions (scheduled games) → {out}")
    return df_features


if __name__ == "__main__":
    predict_batch()
