# ============================================================
# 🏀 NBA Analytics v3
# Module: Totals Model Training (Over/Under)
# File: src/model/train_totals.py
# Author: Sadiq
#
# Description:
#     Trains a regression model to predict game total points
#     (home_score + away_score), using strictly point-in-time
#     correct team-level features built from the canonical
#     long-format data. Only home-team rows are used so each
#     game appears exactly once.
#
#     Persists the model in the registry with rich metadata:
#       - model_type: "totals"
#       - target: "total_points"
#       - market: "ou"
#       - feature_version + feature columns
#       - training date range
#       - train/test metrics (MSE, MAE)
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.config.paths import LONG_SNAPSHOT, MODEL_REGISTRY_DIR
from src.features.builder import FeatureBuilder, FeatureConfig


MODEL_NAME_TOTALS = "rf_totals"


@dataclass
class TotalsTrainingConfig:
    feature_version: str = "v1"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    max_depth: Optional[int] = None
    min_samples_leaf: int = 2
    model_name: str = MODEL_NAME_TOTALS


def _load_long() -> pd.DataFrame:
    if not LONG_SNAPSHOT.exists():
        raise FileNotFoundError(f"Long-format snapshot not found at {LONG_SNAPSHOT}")
    df = pd.read_parquet(LONG_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _build_totals_training_frame(cfg: TotalsTrainingConfig) -> pd.DataFrame:
    """
    Builds a game-level training dataframe for totals:
      - uses FeatureBuilder(features vX)
      - merges back 'points_for' and 'points_against'
      - restricts to home rows (is_home==True)
      - target = points_for + points_against
    """
    df_long = _load_long()

    fb = FeatureBuilder(config=FeatureConfig(version=cfg.feature_version))
    features = fb.build_from_long(df_long=df_long)

    # Merge back points_for / points_against / is_home
    df_merged = features.merge(
        df_long[["game_id", "team", "date", "points_for", "points_against", "is_home"]],
        on=["game_id", "team", "date"],
        how="left",
    )

    # Keep only home rows so each game_id appears once
    df_home = df_merged[df_merged["is_home"] == True].copy()  # noqa: E712

    df_home["total_points"] = df_home["points_for"] + df_home["points_against"]

    missing_target = df_home["total_points"].isna().sum()
    if missing_target > 0:
        logger.warning(f"{missing_target} rows missing total_points; dropping them.")
        df_home = df_home.dropna(subset=["total_points"])

    return df_home


def _time_based_split(df: pd.DataFrame, cfg: TotalsTrainingConfig):
    df = df.sort_values("date")
    unique_dates = sorted(df["date"].unique())
    if len(unique_dates) < 2:
        raise ValueError(
            "Not enough dates to perform a time-based split for totals model."
        )

    cutoff_index = int(len(unique_dates) * (1 - cfg.test_size))
    cutoff_date = unique_dates[cutoff_index]

    train_df = df[df["date"] < cutoff_date].copy()
    test_df = df[df["date"] >= cutoff_date].copy()

    if train_df.empty or test_df.empty:
        logger.warning(
            "Time-based split failed; falling back to random split for totals model."
        )
        features = [
            c
            for c in df.columns
            if c
            not in ("total_points", "date", "game_id", "team", "opponent", "is_home")
        ]
        X = df[features]
        y = df["total_points"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state
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

    feature_cols = [
        c
        for c in train_df.columns
        if c not in ("total_points", "date", "game_id", "team", "opponent", "is_home")
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["total_points"]
    X_test = test_df[feature_cols]
    y_test = test_df["total_points"]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_cols,
        df["date"].min(),
        df["date"].max(),
    )


def _save_model(
    model,
    cfg: TotalsTrainingConfig,
    feature_version: str,
    feature_cols: list[str],
    train_start_date,
    train_end_date,
    train_mae: float,
    train_rmse: float,
    test_mae: float,
    test_rmse: float,
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
        "model_type": "totals",
        "target": "total_points",
        "market": "ou",
        "version": timestamp,
        "path": str(model_path),
        "created_at_utc": datetime.utcnow().isoformat(),
        "feature_version": feature_version,
        "feature_cols": feature_cols,
        "train_start_date": train_start_date.isoformat() if train_start_date else None,
        "train_end_date": train_end_date.isoformat() if train_end_date else None,
        "train_mae": float(train_mae),
        "train_rmse": float(train_rmse),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "params": {
            "n_estimators": cfg.n_estimators,
            "max_depth": cfg.max_depth,
            "min_samples_leaf": cfg.min_samples_leaf,
            "random_state": cfg.random_state,
        },
        "is_production": False,
    }

    Path(metadata_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.success(f"[Totals] Model saved → {model_path}")
    logger.info(f"[Totals] Model metadata saved → {metadata_path}")

    if index_path.exists():
        registry = json.loads(index_path.read_text())
    else:
        registry = {"models": []}

    registry["models"].append(metadata)
    index_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    logger.info(f"[Totals] Model registry index updated → {index_path}")


def train_totals_model(cfg: Optional[TotalsTrainingConfig] = None):
    cfg = cfg or TotalsTrainingConfig()
    logger.info(f"🏀 Starting totals model training with config: {asdict(cfg)}")

    df_train = _build_totals_training_frame(cfg)
    if df_train.empty:
        raise RuntimeError("No training data available for totals model.")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_cols,
        start_date,
        end_date,
    ) = _time_based_split(df_train, cfg)

    logger.info(
        f"[Totals] Training on {len(X_train)} samples, testing on {len(X_test)} samples "
        f"from {start_date} to {end_date}"
    )

    model = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

    logger.info(f"[Totals] Training MAE:  {train_mae:.3f}")
    logger.info(f"[Totals] Training RMSE: {train_rmse:.3f}")
    logger.info(f"[Totals] Test MAE:      {test_mae:.3f}")
    logger.info(f"[Totals] Test RMSE:     {test_rmse:.3f}")

    _save_model(
        model=model,
        cfg=cfg,
        feature_version=cfg.feature_version,
        feature_cols=feature_cols,
        train_start_date=start_date,
        train_end_date=end_date,
        train_mae=train_mae,
        train_rmse=train_rmse,
        test_mae=test_mae,
        test_rmse=test_rmse,
    )

    logger.success("🏀 Totals model training complete.")
    return model, {
        "train_mae": float(train_mae),
        "train_rmse": float(train_rmse),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "feature_version": cfg.feature_version,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    train_totals_model()
