from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Save Model
# File: src/model/registry/save_model.py
# Author: Sadiq
#
# Description:
#     Save a trained model artifact and register its metadata
#     in the model registry. Version-agnostic and aligned with
#     the modern registry structure.
# ============================================================

import joblib
from datetime import datetime
from pathlib import Path
from loguru import logger

from src.config.paths import MODEL_DIR
from src.model.registry.model_registry import register_model, ModelMeta


def save_model(
    model,
    model_type: str,
    version: str,
    feature_version: str | None,
    metrics: dict | None,
    train_start_date: str | None,
    train_end_date: str | None,
) -> ModelMeta:
    """
    Save a trained model and register it in the model registry.

    Args:
        model: trained model object
        model_type: "moneyline" | "totals" | "spread"
        version: version string (semantic, date-based, etc.)
        feature_version: optional feature schema version
        metrics: dict of training metrics
        train_start_date: ISO date string
        train_end_date: ISO date string

    Returns:
        ModelMeta object
    """

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------
    if metrics is None:
        metrics = {}

    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dictionary")

    if train_end_date is None:
        raise ValueError("train_end_date must be provided")

    # --------------------------------------------------------
    # Build artifact path
    # --------------------------------------------------------
    model_name = f"{model_type}_{version}_{train_end_date}"
    artifact_rel = Path(model_type) / f"{model_name}.joblib"
    artifact_path = MODEL_DIR / artifact_rel

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)

    logger.info(f"Saved model artifact → {artifact_path}")

    # --------------------------------------------------------
    # Build metadata (timestamps auto-generated)
    # --------------------------------------------------------
    meta = ModelMeta(
        model_type=model_type,
        version=version,
        model_name=model_name,
        artifact_path=str(artifact_rel),
        feature_version=feature_version,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        metrics=metrics,
        is_production=False,
    )

    # --------------------------------------------------------
    # Register model
    # --------------------------------------------------------
    register_model(meta)

    return meta