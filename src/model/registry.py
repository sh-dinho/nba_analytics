# ============================================================
# 🏀 NBA Analytics v3
# Module: Model Registry
# File: src/model/registry.py
# Author: Sadiq
#
# Description:
#     Unified model registry for:
#       - Moneyline models
#       - Totals models
#       - Spread models
#
#     Supports:
#       - Loading latest model by type
#       - Loading latest production model
#       - Listing all models
#       - Saving new model metadata
#       - Backwards compatibility with older helper names
#
#     Registry structure:
#       data/models/registry/index.json
#       data/models/registry/<model_name>/<model_name>_<version>.pkl
#       data/models/registry/<model_name>/<model_name>_<version>.json
# ============================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from loguru import logger

from src.config.paths import MODEL_REGISTRY_DIR


# ------------------------------------------------------------
# Registry index helpers
# ------------------------------------------------------------

INDEX_PATH = MODEL_REGISTRY_DIR / "index.json"


def _load_registry_index() -> dict:
    """Load the model registry index.json."""
    if not INDEX_PATH.exists():
        logger.warning(
            f"Model registry index not found at {INDEX_PATH}. Creating new index."
        )
        return {"models": []}

    try:
        return json.loads(INDEX_PATH.read_text())
    except Exception as e:
        logger.error(f"Failed to read registry index: {e}")
        return {"models": []}


def _save_registry_index(registry: dict):
    """Write the registry index back to disk."""
    INDEX_PATH.write_text(json.dumps(registry, indent=2))
    logger.info(f"Registry index updated → {INDEX_PATH}")


# ------------------------------------------------------------
# Query helpers
# ------------------------------------------------------------


def list_models(model_type: Optional[str] = None) -> List[dict]:
    """Return all models, optionally filtered by model_type."""
    registry = _load_registry_index()
    models = registry.get("models", [])

    if model_type:
        models = [m for m in models if m.get("model_type") == model_type]

    return models


def get_latest_model_metadata(model_type: str) -> dict:
    """
    Return metadata for the latest model of a given type.
    Prefers production models if available.
    """
    models = list_models(model_type)

    if not models:
        raise RuntimeError(f"No models of type '{model_type}' found in registry.")

    prod = [m for m in models if m.get("is_production")]
    candidates = prod or models

    latest = sorted(candidates, key=lambda m: m["created_at_utc"])[-1]

    logger.info(
        f"[Registry] Latest {model_type} model → "
        f"{latest['model_name']} v{latest['version']} (prod={latest.get('is_production')})"
    )
    return latest


def load_model_and_metadata(model_type: str) -> Tuple[Any, dict]:
    """
    Load the latest model (preferring production) and return (model, metadata).
    """
    meta = get_latest_model_metadata(model_type)
    model_path = Path(meta["path"])

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = pd.read_pickle(model_path)
    return model, meta


# Backwards compatibility alias
load_latest_model_metadata = get_latest_model_metadata
_load_model_and_meta = load_model_and_metadata
_load_model_and_metadata = load_model_and_metadata


# ------------------------------------------------------------
# Saving new models
# ------------------------------------------------------------


def save_new_model_metadata(metadata: dict):
    """
    Append a new model metadata entry to the registry index.
    """
    registry = _load_registry_index()
    registry.setdefault("models", []).append(metadata)
    _save_registry_index(registry)
    logger.success(
        f"[Registry] Registered model {metadata['model_name']} v{metadata['version']}"
    )


# ------------------------------------------------------------
# Production model management
# ------------------------------------------------------------


def set_production_model(model_name: str, version: str):
    """
    Mark a specific model version as production and unset others of same type.
    """
    registry = _load_registry_index()
    updated = False

    for m in registry.get("models", []):
        if m["model_name"] == model_name:
            m["is_production"] = m["version"] == version
            updated = True

    if not updated:
        raise RuntimeError(
            f"Model {model_name} version {version} not found in registry."
        )

    _save_registry_index(registry)
    logger.success(f"[Registry] Set {model_name} v{version} as production.")


def get_production_model(model_type: str) -> Optional[dict]:
    """Return the production model metadata for a given type."""
    models = list_models(model_type)
    prod = [m for m in models if m.get("is_production")]

    if not prod:
        return None

    return prod[-1]
