from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Load Model
# File: src/model/registry/load_model.py
# Author: Sadiq
#
# Description:
#     Thin wrapper around the registry to load a model by
#     model_type and version. Version-agnostic and aligned
#     with the modern registry structure.
# ============================================================

import joblib
from pathlib import Path
from loguru import logger

from src.model.registry.model_registry import load_registry, ModelMeta
from src.config.paths import MODEL_DIR


def load_model(model_type: str, version: str):
    """
    Load a specific model version for a given model_type.

    Args:
        model_type: "moneyline" | "totals" | "spread"
        version: version string (semantic, date-based, etc.)

    Returns:
        Loaded model object.
    """
    registry = load_registry()
    models = [
        m for m in registry.get("models", [])
        if m.get("model_type") == model_type and m.get("version") == version
    ]

    if not models:
        raise ValueError(
            f"No model found for model_type={model_type}, version={version}"
        )

    # Sort by created_at_utc if multiple entries exist
    def sort_key(m):
        return m.get("created_at_utc", "")

    latest = sorted(models, key=sort_key)[-1]
    meta = ModelMeta(**latest)

    artifact_path = MODEL_DIR / meta.artifact_path

    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {artifact_path}")

    logger.info(
        f"Loading model: {meta.model_name} "
        f"({model_type}) v{version} from {artifact_path}"
    )

    return joblib.load(artifact_path)