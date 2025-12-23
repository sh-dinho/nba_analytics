from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Totals Model Training (Over/Under)
# File: src/model/train_totals.py
# Author: Sadiq
#
# Description:
#     Trains a regression model to predict total game points
#     (score + opponent_score) using strictly point-in-time
#     correct team-game features from the canonical v4
#     ingestion pipeline. Only home rows are used so each
#     game appears exactly once.
#
#     Persists the model in the v4 registry with metadata:
#       - model_type: "totals"
#       - target: "total_points"
#       - market: "ou"
#       - feature_version + feature columns
#       - training date range
#       - MAE / RMSE metrics
# ============================================================

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.config.paths import LONG_SNAPSHOT, MODEL_DIR
from src.features.builder import FeatureBuilder
from src.model.registry import register_model
from src.model.training_core import _drop_non_numeric_features, _fill_missing_values


# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------


@dataclass
class TotalsTrainingConfig:
    feature_version: str = "v4"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    max_depth: Optional[int] = None
    min_samples_leaf: int = 2


# ------------------------------------------------------------
# Load canonical long snapshot
# ------------------------------------------------------------


def _load_long() -> pd.DataFrame:
    df = pd.read_parquet(LONG_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"])  # FIXED: keep datetime64[ns]
    return df


# ------------------------------------------------------------
# Build features + target (total_points)
# ------------------------------------------------------------


def _build_training_frame(cfg: TotalsTrainingConfig) -> pd.DataFrame:
    df = _load_long()

    fb = FeatureBuilder(version=cfg.feature_version)
    features = fb.build_from_long(df)

    # Only home rows → one row per game
    df_home = df[df["is_home"] == 1].copy()
    df_home["total_points"] = df_home["score"] + df_home["opponent_score"]

    merged = features.merge(
        df_home[["game_id", "team", "date", "total_points"]],
        on=["game_id", "team", "date"],
        how="inner",
    )

    merged = merged.dropna(subset=["total_points"])
    return merged


# ------------------------------------------------------------
# Time-based split
# ------------------------------------------------------------


def _time_split(df: pd.DataFrame, cfg: TotalsTrainingConfig):
    df = df.sort_values("date")
    unique_dates = sorted(df["date"].unique())

    cutoff_idx = int(len(unique_dates) * (1 - cfg.test_size))
    cutoff_date = unique_dates[cutoff_idx]

    train_df = df[df["date"] < cutoff_date]
    test_df = df[df["date"] >= cutoff_date]

    feature_cols = [
        c for c in df.columns if c not in ("total_points", "date", "game_id", "team")
    ]

    return (
        train_df[feature_cols],
        test_df[feature_cols],
        train_df["total_points"],
        test_df["total_points"],
        feature_cols,
        df["date"].min(),
        df["date"].max(),
    )


# ------------------------------------------------------------
# Save model + metadata
# ------------------------------------------------------------


def _save_model(
    model,
    feature_cols: List[str],
    start_date,
    end_date,
    train_mae,
    test_mae,
    train_rmse,
    test_rmse,
):
    version = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    out_dir = MODEL_DIR / "totals"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{version}.pkl"
    pd.to_pickle(model, model_path)

    meta = {
        "model_type": "totals",
        "version": version,
        "created_at": datetime.utcnow().isoformat(),
        "is_production": False,
        "feature_version": "v4",
        "feature_cols": feature_cols,
        "metrics": {
            "target": "total_points",
            "market": "ou",
            "train_mae": float(train_mae),
            "test_mae": float(test_mae),
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse),
            "train_start_date": str(start_date),
            "train_end_date": str(end_date),
        },
    }

    register_model(meta)
    logger.success(f"[Totals] Model saved → {model_path}")
    logger.success(f"[Totals] Model registered → version={version}")


# ------------------------------------------------------------
# Main training entry point
# ------------------------------------------------------------


def train_totals_model(cfg: Optional[TotalsTrainingConfig] = None):
    cfg = cfg or TotalsTrainingConfig()
    logger.info("🏀 Starting totals model training")

    df = _build_training_frame(cfg)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_cols,
        start_date,
        end_date,
    ) = _time_split(df, cfg)

    # Clean features
    X_train = _drop_non_numeric_features(X_train)
    X_test = _drop_non_numeric_features(X_test)
    X_train = _fill_missing_values(X_train)
    X_test = _fill_missing_values(X_test)

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

    _save_model(
        model,
        feature_cols,
        start_date,
        end_date,
        train_mae,
        test_mae,
        train_rmse,
        test_rmse,
    )

    logger.success("🏀 Totals model training complete.")
    return model


if __name__ == "__main__":
    train_totals_model()
