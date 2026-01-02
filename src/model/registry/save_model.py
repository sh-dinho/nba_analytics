from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Save Model
# File: src/model/registry/save_model.py
# Author: Sadiq
# ============================================================

import joblib
from pathlib import Path
from loguru import logger

from src.config.paths import MODEL_DIR
from src.model.registry import register_model, ModelMeta


def save_model(
    model,
    model_type: str,
    version: str | int,
    metrics: dict,
    feature_list: list[str],
    feature_version: str | None = None,
    model_family: str | None = None,
    hyperparams: dict | None = None,
    train_start_date: str | None = None,
    train_end_date: str | None = None,
) -> ModelMeta:
    """
    Save a trained model artifact and register it in the model registry.
    """

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------
    version = str(version)

    if metrics is None or not isinstance(metrics, dict):
        raise ValueError("metrics must be a non-empty dictionary")

    if train_end_date is None:
        raise ValueError("train_end_date must be provided")

    if hyperparams is None:
        hyperparams = {}

    if feature_version is None:
        feature_version = "unknown"

    # --------------------------------------------------------
    # Sanitize timestamp for Windows-safe filenames
    # --------------------------------------------------------
    safe_end_date = (
        str(train_end_date)
        .replace(":", "-")
        .replace(" ", "_")
    )

    # --------------------------------------------------------
    # Normalize date fields for JSON serialization
    # --------------------------------------------------------
    if hasattr(train_start_date, "isoformat"):
        train_start_date = train_start_date.isoformat()

    if hasattr(train_end_date, "isoformat"):
        train_end_date = train_end_date.isoformat()

    # --------------------------------------------------------
    # Build artifact path
    # --------------------------------------------------------
    model_name = f"{model_type}_{version}_{safe_end_date}"
    artifact_rel = Path(model_type) / f"{model_name}.joblib"
    artifact_path = MODEL_DIR / artifact_rel

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)

    logger.info(f"Saved model artifact → {artifact_path}")

    # --------------------------------------------------------
    # Build metadata object
    # --------------------------------------------------------
    meta = ModelMeta(
        model_type=model_type,
        version=version,
        model_name=model_name,
        artifact_path=str(artifact_rel.as_posix()),
        model_family=model_family,
        hyperparams=hyperparams,
        feature_version=str(feature_version),
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        metrics=metrics,
        is_production=False,
    )

    # --------------------------------------------------------
    # Register model
    # --------------------------------------------------------
    register_model(meta)

    logger.success(
        f"Registered model: {meta.model_name} "
        f"({model_type}) v{version} | family={model_family}"
    )

    return meta