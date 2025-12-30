from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List

import joblib
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from src.config.paths import MODEL_DIR
from src.model.registry import register_model
from src.model.schema import SchemaSaver


# ============================================================
# 🏀 NBA Analytics v4
# Module: Training Core
# File: src/model/training_core.py
# Author: Sadiq
#
# Description:
#     Trains models, evaluates them, saves artifacts, and
#     registers metadata in the v4 model registry.
#
#     v4 alignment:
#       - Uses FeatureBuilder v4 outputs (upstream)
#       - Auto-derives moneyline targets from scores if missing
#       - Persists sklearn models under MODEL_DIR
#       - Registers feature_version + feature_cols for prediction
#       - Drops non-numeric columns
#       - Fills NaN values with 0 (safe for ML models)
#       - Saves training-time schema using SchemaSaver
# ============================================================


# ------------------------------------------------------------
# Model Metadata
# ------------------------------------------------------------


@dataclass
class ModelMetadata:
    model_type: str
    version: str
    created_at: str
    is_production: bool
    metrics: Dict[str, Any]
    feature_version: str
    feature_cols: List[str]


# ------------------------------------------------------------
# Target selection + auto-generation
# ------------------------------------------------------------


def _select_target_column(model_type: str) -> str:
    mapping = {
        "moneyline": "moneyline_win",
        "totals": "totals_over",
        "spread": "spread_cover",
    }
    if model_type not in mapping:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return mapping[model_type]


def _ensure_target_column(model_type: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the target column exists.

    For moneyline:
      - If `moneyline_win` missing, derive it from score vs opponent_score.
    For totals/spread:
      - Require explicit target (future odds ingestion will populate these).
    """
    target_col = _select_target_column(model_type)

    if target_col in df.columns:
        return df

    if model_type == "moneyline":
        logger.warning("[TrainingCore] moneyline_win missing — generating from scores.")
        df = df.copy()
        df["moneyline_win"] = (df["score"] > df["opponent_score"]).astype(int)
        return df

    raise ValueError(f"Missing target column: {target_col}")


# ------------------------------------------------------------
# Feature cleaning — enforce numeric-only + fill NaNs
# ------------------------------------------------------------


def _drop_non_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = X.select_dtypes(include=["number"]).columns
    dropped = [c for c in X.columns if c not in numeric_cols]

    if dropped:
        logger.warning(
            f"[TrainingCore] Dropping non-numeric feature columns: {dropped}"
        )

    return X[numeric_cols]


def _fill_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """
    Replace NaN values with 0 for all numeric features.
    This matches v3 behavior and ensures LogisticRegression compatibility.
    """
    missing_before = X.isna().sum().sum()
    if missing_before > 0:
        logger.warning(
            f"[TrainingCore] Filling {missing_before} missing values with 0."
        )
    return X.fillna(0)


# ------------------------------------------------------------
# Feature matrix + target extraction
# ------------------------------------------------------------


def _get_feature_matrix_and_target(
    model_type: str,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, str]:
    df = _ensure_target_column(model_type, df)
    target_col = _select_target_column(model_type)

    drop_cols = ["game_id", "team", "opponent", "date", "season", target_col]
    existing_drop = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=existing_drop)

    # Clean features
    X = _drop_non_numeric_features(X)
    X = _fill_missing_values(X)

    y = df[target_col]

    return X, y, target_col


# ------------------------------------------------------------
# Model training + evaluation
# ------------------------------------------------------------


def _train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    return model


def _evaluate_model(
    model: LogisticRegression,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, float]:
    if len(y_val.unique()) < 2:
        logger.warning(
            "[TrainingCore] Validation set has only one class; "
            "log_loss may be ill-defined."
        )

    probs = model.predict_proba(X_val)[:, 1]
    return {"log_loss": float(log_loss(y_val, probs))}


# ------------------------------------------------------------
# Artifact saving
# ------------------------------------------------------------


def _save_model_artifact(
    model: Any,
    model_type: str,
    version: str,
) -> None:
    """
    Persist model artifact under MODEL_DIR / <type> / <version>.pkl
    """
    out_dir = MODEL_DIR / model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{version}.pkl"
    joblib.dump(model, path)
    logger.info(f"[TrainingCore] Model artifact saved → {path}")


# ------------------------------------------------------------
# Main training entry point
# ------------------------------------------------------------


def train_and_register_model(
    model_type: str,
    df: pd.DataFrame,
    feature_version: str,
) -> ModelMetadata:
    """
    Train a model of given type on the provided feature DataFrame,
    evaluate it, save the artifact, save the training schema,
    and register metadata.

    Returns the ModelMetadata for convenience.
    """
    logger.info(f"[TrainingCore] Training model_type={model_type}")

    if df.empty:
        raise ValueError("[TrainingCore] Received empty DataFrame for training.")

    # --------------------------------------------------------
    # Prepare X, y
    # --------------------------------------------------------
    X, y, target_col = _get_feature_matrix_and_target(model_type, df)

    feature_cols = list(X.columns)
    logger.debug(
        f"[TrainingCore] Using {len(feature_cols)} features for {model_type} "
        f"(target={target_col})"
    )

    # --------------------------------------------------------
    # Train/validation split
    # --------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------
    model = _train_logistic_regression(X_train, y_train)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    metrics = _evaluate_model(model, X_val, y_val)
    logger.info(f"[TrainingCore] Validation metrics={metrics}")

    # Version stamp
    version = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    # --------------------------------------------------------
    # Save artifact
    # --------------------------------------------------------
    _save_model_artifact(model, model_type, version)

    # --------------------------------------------------------
    # Save training schema (FIXED: Using SchemaSaver)
    # --------------------------------------------------------
    out_dir = MODEL_DIR / model_type
    schema_path = out_dir / f"{version}_schema.json"

    # Using the centralized saver from src.model.schema
    SchemaSaver(
        model_type=model_type, model_version=version, feature_version=feature_version
    ).save(X, schema_path)

    # --------------------------------------------------------
    # Register model metadata
    # --------------------------------------------------------
    meta = ModelMetadata(
        model_type=model_type,
        version=version,
        created_at=datetime.utcnow().isoformat(),
        is_production=False,
        metrics=metrics,
        feature_version=feature_version,
        feature_cols=feature_cols,
    )

    register_model(meta)
    logger.info(
        f"[TrainingCore] Registered model_type={model_type}, "
        f"version={version}, production={meta.is_production}"
    )

    return meta
