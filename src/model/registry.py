from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Model Registry
# File: src/model/registry.py
# Author: Sadiq
#
# Description:
#     Central registry for all models (moneyline, totals,
#     spread_regression, spread_classification).
#
#     Features:
#       - Persistent JSON index (v4 schema)
#       - Backward compatible register_model(meta)
#       - Load model + metadata by type/version/production flag
#       - List models
#       - Promote model to production (enforce single prod per type)
#       - Delete model + artifact
# ============================================================

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import pandas as pd
from loguru import logger

from src.config.paths import MODEL_DIR, MODEL_REGISTRY_PATH, MODEL_REGISTRY_INDEX

# ------------------------------------------------------------
# Registry schema / constants
# ------------------------------------------------------------

# Registry JSON structure:
# {
#   "models": [
#     {
#       "model_type": "moneyline",
#       "version": "20251223184546",
#       "created_at": "...",
#       "is_production": false,
#       "feature_version": "v4",
#       "feature_cols": [...],
#       "metrics": {...}
#     },
#     ...
#   ]
# }

REGISTRY_ROOT: Path = MODEL_REGISTRY_PATH.parent


# ------------------------------------------------------------
# Low-level IO helpers
# ------------------------------------------------------------


def _ensure_registry_dir() -> None:
    REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)


def _load_registry() -> Dict[str, Any]:
    _ensure_registry_dir()

    if not MODEL_REGISTRY_PATH.exists():
        logger.warning(
            f"Registry index not found at {MODEL_REGISTRY_PATH} — creating new registry."
        )
        return {"models": []}

    try:
        with MODEL_REGISTRY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load registry index; resetting. Error: {e}")
        return {"models": []}

    # Basic shape validation
    if (
        not isinstance(data, dict)
        or "models" not in data
        or not isinstance(data["models"], list)
    ):
        logger.error("Registry structure invalid — resetting registry.")
        return {"models": []}

    return data


def _save_registry(index: Dict[str, Any]) -> None:
    _ensure_registry_dir()
    with MODEL_REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    logger.info(f"Registry updated → {MODEL_REGISTRY_PATH}")


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------


def _meta_to_dict(meta: Any) -> Dict[str, Any]:
    """
    Normalize metadata into a plain dict.

    Accepts:
      - dataclass instances (training_core moneyline)
      - plain dicts (totals, spread trainers)
    """
    if is_dataclass(meta):
        return asdict(meta)
    if isinstance(meta, dict):
        return meta
    raise TypeError(f"Unsupported metadata type: {type(meta)}")


def _filter_models(
    index: Dict[str, Any],
    model_type: Optional[str] = None,
    production_only: bool = False,
) -> List[Dict[str, Any]]:
    models = index.get("models", [])

    if model_type is not None:
        models = [m for m in models if m.get("model_type") == model_type]

    if production_only:
        models = [m for m in models if m.get("is_production")]

    return models


def _select_latest(models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Select latest model by version (timestamp string) or created_at as fallback.
    """
    if not models:
        raise ValueError("No models available to select latest from.")

    # Prefer version (timestamp-like); fallback to created_at
    def key_fn(m: Dict[str, Any]) -> str:
        v = m.get("version") or m.get("created_at") or ""
        return str(v)

    return max(models, key=key_fn)


def _artifact_path(model_type: str, version: str) -> Path:
    """
    Artifact path convention:
      MODEL_DIR / <model_type> / <version>.pkl
    """
    return MODEL_DIR / model_type / f"{version}.pkl"


# ------------------------------------------------------------
# Public API: Registering models
# ------------------------------------------------------------


def register_model(meta: Any) -> None:
    """
    Register a model in the v4 registry.

    Accepts:
      - dataclass instances (moneyline via training_core)
      - plain dicts (totals, spread trainers)

    Behavior:
      - Normalizes metadata to dict
      - Appends to registry
      - Does NOT automatically change production flags
        (use promote_model for that)
    """
    index = _load_registry()
    meta_dict = _meta_to_dict(meta)

    # Basic sanity checks
    required_fields = ["model_type", "version"]
    missing = [f for f in required_fields if f not in meta_dict]
    if missing:
        raise ValueError(f"Metadata missing required fields: {missing}")

    if "is_production" not in meta_dict:
        meta_dict["is_production"] = False

    index.setdefault("models", []).append(meta_dict)
    _save_registry(index)

    logger.success(
        f"Registered model version={meta_dict['version']} "
        f"type={meta_dict['model_type']}"
    )


# ------------------------------------------------------------
# Public API: Listing models
# ------------------------------------------------------------


def list_models(
    model_type: Optional[str] = None,
    production_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    List models in the registry.

    Args:
        model_type: Optional filter by type (e.g., "moneyline").
        production_only: If True, only return models with is_production=True.

    Returns:
        List of metadata dicts.
    """
    index = _load_registry()
    models = _filter_models(
        index, model_type=model_type, production_only=production_only
    )
    return models


# ------------------------------------------------------------
# Public API: Model promotion (production flag enforcement)
# ------------------------------------------------------------


def promote_model(model_type: str, version: str) -> None:
    """
    Mark a specific model as production and demote all other models
    of the same type.

    Args:
        model_type: Model type (e.g., "moneyline", "totals").
        version: Version string to promote.
    """
    index = _load_registry()
    models = index.get("models", [])

    found = False
    for m in models:
        if m.get("model_type") != model_type:
            continue
        if m.get("version") == version:
            m["is_production"] = True
            found = True
        else:
            m["is_production"] = False

    if not found:
        raise ValueError(
            f"No model found to promote for type={model_type}, version={version}"
        )

    _save_registry(index)
    logger.success(f"Promoted model type={model_type}, version={version} to production")


# ------------------------------------------------------------
# Public API: Model deletion
# ------------------------------------------------------------


def delete_model(model_type: str, version: str) -> None:
    """
    Delete a model from the registry and remove its artifact file.

    Args:
        model_type: Model type.
        version: Version string.
    """
    index = _load_registry()
    models = index.get("models", [])

    before = len(models)
    models = [
        m
        for m in models
        if not (m.get("model_type") == model_type and m.get("version") == version)
    ]
    removed = before - len(models)

    if removed == 0:
        logger.warning(
            f"No model found to delete for type={model_type}, version={version}"
        )
    else:
        logger.info(f"Deleted registry entry for type={model_type}, version={version}")

    index["models"] = models
    _save_registry(index)

    # Remove artifact file
    path = _artifact_path(model_type, version)
    if path.exists():
        try:
            path.unlink()
            logger.info(f"Deleted model artifact → {path}")
        except Exception as e:
            logger.error(f"Failed to delete model artifact {path}: {e}")


# ------------------------------------------------------------
# Public API: Load model + metadata
# ------------------------------------------------------------


def load_model_and_metadata(
    model_type: str,
    version: Optional[str] = None,
    production_only: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a model artifact and its metadata.

    Args:
        model_type: Model type to load.
        version: Optional specific version. If None:
            - if production_only=True → load latest production model
            - else → load latest model by version
        production_only: Whether to restrict search to production models.

    Returns:
        (model, metadata_dict)
    """
    index = _load_registry()
    candidates = _filter_models(
        index, model_type=model_type, production_only=production_only
    )

    if not candidates:
        raise ValueError(f"No models found for type={model_type}")

    if version is not None:
        # Filter by requested version
        candidates = [m for m in candidates if m.get("version") == version]
        if not candidates:
            raise ValueError(
                f"No models found for type={model_type}, version={version}"
            )
        meta = candidates[0]
    else:
        # Select latest (by version or created_at)
        meta = _select_latest(candidates)

    model_version = meta["version"]
    path = _artifact_path(model_type, model_version)

    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path} for type={model_type}, version={model_version}"
        )

    model = pd.read_pickle(path)
    logger.info(f"Loaded model type={model_type}, version={model_version} from {path}")
    return model, meta


def _load_index() -> dict:
    """
    Load the model registry index JSON file.
    Returns an empty structure if missing or corrupted.
    """
    if not MODEL_REGISTRY_INDEX.exists():
        logger.warning(f"[Registry] Index not found: {MODEL_REGISTRY_INDEX}")
        return {"models": []}

    try:
        with open(MODEL_REGISTRY_INDEX, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"[Registry] Failed to load index: {e}")
        return {"models": []}
