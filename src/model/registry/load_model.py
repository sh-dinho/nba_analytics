from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Load Model
# File: src/model/registry/load_model.py
# Author: Sadiq
#
# Description:
#     Load a specific model version from the upgraded registry.
#     Fully aligned with the modern registry structure.
# ============================================================

import joblib
from pathlib import Path
from datetime import datetime
from loguru import logger

from src.model.registry import load_registry, ModelMeta
from src.config.paths import MODEL_DIR


def load_model(model_type: str, version: int | str):
    """
    Load a specific model version for a given model_type.

    Args:
        model_type: "moneyline" | "totals" | "spread"
        version: integer or string version identifier

    Returns:
        model: loaded model object
        meta: ModelMeta containing registry metadata
    """
    version = str(version)  # normalize

    registry = load_registry()

    if "models" not in registry:
        raise RuntimeError("Model registry is missing 'models' key.")

    # Filter entries
    candidates = [
        m for m in registry["models"]
        if m.get("model_type") == model_type and str(m.get("version")) == version
    ]

    if not candidates:
        raise ValueError(
            f"No model found for model_type='{model_type}', version='{version}'."
        )

    # --------------------------------------------------------
    # Enhancement 1: Sort by actual datetime, not string
    # --------------------------------------------------------
    def sort_key(m):
        ts = m.get("created_at_utc")
        try:
            return datetime.fromisoformat(ts) if ts else datetime.min
        except Exception:
            return datetime.min

    latest = sorted(candidates, key=sort_key)[-1]
    meta = ModelMeta(**latest)

    # --------------------------------------------------------
    # Enhancement 2: Use full artifact path from metadata
    # --------------------------------------------------------
    artifact_path = MODEL_DIR / meta.artifact_path

    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {artifact_path}")

    logger.info(
        f"Loading model: {meta.model_name} "
        f"({model_type}) v{version} from {artifact_path}"
    )

    model = joblib.load(artifact_path)
    return model, meta