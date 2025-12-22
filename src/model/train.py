# ============================================================
# 🏀 NBA Analytics v3
# Module: Model Training
# File: src/model/train.py
# Author: Sadiq
#
# Description:
#     Trains a classification model on strictly point-in-time
#     correct team-level features built from the canonical
#     long-format data. Persists the model in a versioned
#     registry with rich metadata, including:
#       - feature version
#       - feature columns
#       - date range
#       - basic metrics (accuracy)
# ============================================================

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import json

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.config.paths import LONG_SNAPSHOT, MODEL_REGISTRY_DIR, DATA_DIR
from src.features.builder import FeatureBuilder, FeatureConfig


# Ensure registry dir exists
MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingConfig:
    feature_version: str = "v1"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    max_depth: Optional[int] = None
    min_samples_leaf: int = 2
    model_name: str = "rf_moneyline"


def _load_long() -> pd.DataFrame:
    if not LONG_SNAPSHOT.exists():
        raise FileNotFoundError(f"Long-format snapshot not found at {LONG_SNAPSHOT}")
    df = pd.read_parquet(LONG_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _build_features(cfg: TrainingConfig) -> pd.DataFrame:
    df_long = _load_long()
    fb = FeatureBuilder(config=FeatureConfig(version=cfg.feature_version))
    features = fb.build_from_long(df_long=df_long)

    # Add target (won) back in by merging with long snapshot
    df_merged = features.merge(
        df_long[["game_id", "team", "won", "date"]],
        on=["game_id", "team", "date"],
        how="left",
    )

    if df_merged["won"].isna().any():
        missing = df_merged[df_merged["won"].isna()]
        logger.warning(
            "Some feature rows are missing targets (won). "
            f"Examples (up to 5):\n{missing.head().to_string(index=False)}"
        )
        df_merged = df_merged.dropna(subset=["won"])

    df_merged["won"] = df_merged["won"].astype(int)

    return df_merged


def _time_based_split(df: pd.DataFrame, cfg: TrainingConfig):
    df = df.sort_values("date")
    unique_dates = sorted(df["date"].unique())
    if len(unique_dates) < 2:
        raise ValueError("Not enough dates to perform a time-based split.")

    # Use the last X% of dates as test set
    cutoff_index = int(len(unique_dates) * (1 - cfg.test_size))
    cutoff_date = unique_dates[cutoff_index]

    train_df = df[df["date"] < cutoff_date].copy()
    test_df = df[df["date"] >= cutoff_date].copy()

    if train_df.empty or test_df.empty:
        # fallback to random split if time-based fails (e.g., small dataset)
        logger.warning("Time-based split failed; falling back to random split.")
        features = [
            c
            for c in df.columns
            if c not in ("won", "date", "game_id", "team", "opponent")
        ]
        X = df[features]
        y = df["won"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
        )
        return (
            X_train,
            X_test,
            y_train,
            y_test,
            features,
            df["date"].min(),
            df["date"].max(),
        )

    features = [
        c
        for c in train_df.columns
        if c not in ("won", "date", "game_id", "team", "opponent")
    ]

    X_train = train_df[features]
    y_train = train_df["won"]
    X_test = test_df[features]
    y_test = test_df["won"]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        features,
        df["date"].min(),
        df["date"].max(),
    )


def _save_model(
    model,
    cfg: TrainingConfig,
    feature_version: str,
    feature_cols: list[str],
    train_start_date,
    train_end_date,
    train_accuracy: float,
    test_accuracy: float,
):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_dir = MODEL_REGISTRY_DIR / cfg.model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{cfg.model_name}_{timestamp}.pkl"
    metadata_path = model_dir / f"{cfg.model_name}_{timestamp}.json"
    index_path = MODEL_REGISTRY_DIR / "index.json"

    pd.to_pickle(model, model_path)

    metadata = {
        "model_name": cfg.model_name,
        "version": timestamp,
        "path": str(model_path),
        "created_at_utc": datetime.utcnow().isoformat(),
        "feature_version": feature_version,
        "feature_cols": feature_cols,
        "train_start_date": train_start_date.isoformat() if train_start_date else None,
        "train_end_date": train_end_date.isoformat() if train_end_date else None,
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "params": {
            "n_estimators": cfg.n_estimators,
            "max_depth": cfg.max_depth,
            "min_samples_leaf": cfg.min_samples_leaf,
            "random_state": cfg.random_state,
        },
        "is_production": False,
    }

    Path(metadata_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.success(f"Model saved → {model_path}")
    logger.info(f"Model metadata saved → {metadata_path}")

    # Update registry index
    if index_path.exists():
        registry = json.loads(index_path.read_text())
    else:
        registry = {"models": []}

    registry["models"].append(metadata)
    index_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    logger.info(f"Model registry index updated → {index_path}")


def train_model(cfg: Optional[TrainingConfig] = None):
    cfg = cfg or TrainingConfig()
    logger.info(f"🏀 Starting model training with config: {asdict(cfg)}")

    df_features = _build_features(cfg)

    X_train, X_test, y_train, y_test, feature_cols, start_date, end_date = (
        _time_based_split(df_features, cfg)
    )

    logger.info(
        f"Training on {len(X_train)} samples, testing on {len(X_test)} samples "
        f"from {start_date} to {end_date}"
    )

    model = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    logger.info(f"Training accuracy: {train_acc:.3f}")
    logger.info(f"Test accuracy: {test_acc:.3f}")

    _save_model(
        model=model,
        cfg=cfg,
        feature_version=cfg.feature_version,
        feature_cols=feature_cols,
        train_start_date=start_date,
        train_end_date=end_date,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
    )

    logger.success("🏀 Model training complete.")
    return model, {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "feature_version": cfg.feature_version,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    train_model()
