from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Model Training
# File: src/model/train.py
# Author: Sadiq
#
# Description:
#     Trains a classification model on point‑in‑time correct
#     team‑game features. Saves the model into the v4 registry
#     with full metadata:
#       - model_type (moneyline / totals / spread)
#       - feature_version
#       - feature columns
#       - date range
#       - metrics
#       - production flag
# ============================================================

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.config.paths import SCHEDULE_SNAPSHOT
from src.features.builder import FeatureBuilder
from src.model.registry import register_model, ModelMetadata


# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------


@dataclass
class TrainingConfig:
    model_type: str = "moneyline"  # moneyline | totals | spread
    model_name: str = "rf_moneyline"
    feature_version: str = "v1"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    max_depth: Optional[int] = None
    min_samples_leaf: int = 2


# ------------------------------------------------------------
# Load canonical team‑game data
# ------------------------------------------------------------


def _load_team_game() -> pd.DataFrame:
    if not SCHEDULE_SNAPSHOT.exists():
        raise FileNotFoundError(f"Canonical schedule not found at {SCHEDULE_SNAPSHOT}")

    df = pd.read_parquet(SCHEDULE_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ------------------------------------------------------------
# Build features + target
# ------------------------------------------------------------


def _build_features(cfg: TrainingConfig) -> pd.DataFrame:
    df = _load_team_game()

    fb = FeatureBuilder(version=cfg.feature_version)
    features = fb.build_from_long(df)

    # Target: win/loss
    df_target = df[["game_id", "team", "date"]].copy()
    df_target["won"] = (df["score"] > df["opponent_score"]).astype(int)

    merged = features.merge(df_target, on=["game_id", "team", "date"], how="left")

    missing = merged[merged["won"].isna()]
    if not missing.empty:
        logger.warning(
            "Some feature rows missing targets. Dropping them.\n"
            f"{missing.head().to_string(index=False)}"
        )
        merged = merged.dropna(subset=["won"])

    merged["won"] = merged["won"].astype(int)
    return merged


# ------------------------------------------------------------
# Time‑based split
# ------------------------------------------------------------


def _time_split(df: pd.DataFrame, cfg: TrainingConfig):
    df = df.sort_values("date")
    unique_dates = sorted(df["date"].unique())

    if len(unique_dates) < 2:
        raise ValueError("Not enough dates for time‑based split.")

    cutoff_idx = int(len(unique_dates) * (1 - cfg.test_size))
    cutoff_date = unique_dates[cutoff_idx]

    train_df = df[df["date"] < cutoff_date]
    test_df = df[df["date"] >= cutoff_date]

    if train_df.empty or test_df.empty:
        logger.warning("Time split failed → falling back to random split.")
        features = [
            c for c in df.columns if c not in ("won", "date", "game_id", "team")
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
        c for c in train_df.columns if c not in ("won", "date", "game_id", "team")
    ]

    return (
        train_df[features],
        test_df[features],
        train_df["won"],
        test_df["won"],
        features,
        df["date"].min(),
        df["date"].max(),
    )


# ------------------------------------------------------------
# Save model to registry
# ------------------------------------------------------------


def _save_model(
    model,
    cfg: TrainingConfig,
    feature_cols,
    start_date,
    end_date,
    train_acc,
    test_acc,
):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    meta = ModelMetadata(
        model_type=cfg.model_type,
        model_name=cfg.model_name,
        version=timestamp,
        created_at_utc=datetime.utcnow().isoformat(),
        feature_version=cfg.feature_version,
        feature_cols=feature_cols,
        path=f"{cfg.model_name}/{cfg.model_name}_{timestamp}.pkl",
        metrics={
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "train_start_date": str(start_date),
            "train_end_date": str(end_date),
        },
        is_production=False,
    )

    # Save model file
    model_path = Path("data/model_registry") / meta.path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(model, model_path)

    # Register metadata
    register_model(meta)

    logger.success(f"Model saved → {model_path}")
    logger.success(f"Model registered → {meta.model_name} ({meta.version})")


# ------------------------------------------------------------
# Main training entry point
# ------------------------------------------------------------


def train_model(cfg: Optional[TrainingConfig] = None):
    cfg = cfg or TrainingConfig()
    logger.info(f"🏀 Starting model training with config:\n{asdict(cfg)}")

    df = _build_features(cfg)
    X_train, X_test, y_train, y_test, feature_cols, start_date, end_date = _time_split(
        df, cfg
    )

    logger.info(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows.")

    model = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    logger.info(f"Train accuracy: {train_acc:.3f}")
    logger.info(f"Test accuracy: {test_acc:.3f}")

    _save_model(
        model=model,
        cfg=cfg,
        feature_cols=feature_cols,
        start_date=start_date,
        end_date=end_date,
        train_acc=train_acc,
        test_acc=test_acc,
    )

    logger.success("🏀 Training complete.")
    return model


if __name__ == "__main__":
    train_model()
