from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Model Registry
# File: src/model/registry.py
#
# Description:
#     Central registry for all models (moneyline, totals,
#     spread_regression, spread_classification).
#
#     Features:
#       - Persistent JSON index (v4 schema)
#       - Register model metadata
#       - Load model + metadata by type/version/production flag
#       - List models
#       - Promote model to production (enforce single prod per type)
#       - Delete model + artifact
#
#     Cleaned:
#       - Removed dead code
#       - Removed unused imports
#       - Removed MODEL_REGISTRY_INDEX
#       - Removed duplicate index loaders
#       - Unified internal helpers
#       - Ensured consistent naming + structure
# ============================================================

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import pandas as pd
from loguru import logger

from src.config.paths import MODEL_DIR, MODEL_REGISTRY_PATH


# ------------------------------------------------------------
# Unified model wrapper
# ------------------------------------------------------------


class ModelWrapper:
    """
    v4 unified model interface.
    Ensures all loaded models expose predict() and optionally predict_proba().
    """

    def __init__(self, model: Any):
        self.model = model

    def predict(self, X):
        if hasattr(self.model, "predict"):
            return self.model.predict(X)

        if hasattr(self.model, "shape") and getattr(self.model, "shape", [0])[0] == len(
            X
        ):
            logger.warning(
                "[ModelWrapper] Treating underlying array as precomputed predictions."
            )
            return self.model

        if isinstance(self.model, (list, tuple)) and len(self.model) == len(X):
            logger.warning(
                "[ModelWrapper] Treating underlying sequence as precomputed predictions."
            )
            return pd.Series(self.model).values

        raise TypeError(f"Unsupported model type for prediction: {type(self.model)}")

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError(
            f"Underlying model has no predict_proba: {type(self.model)}"
        )


# ------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------


def _ensure_registry_dir() -> None:
    MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_registry() -> Dict[str, Any]:
    _ensure_registry_dir()

    if not MODEL_REGISTRY_PATH.exists():
        logger.warning(
            f"Registry not found → creating new registry at {MODEL_REGISTRY_PATH}"
        )
        return {"models": []}

    try:
        with MODEL_REGISTRY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load registry; resetting. Error: {e}")
        return {"models": []}

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


def _meta_to_dict(meta: Any) -> Dict[str, Any]:
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
    if not models:
        raise ValueError("No models available to select latest from.")

    def key_fn(m: Dict[str, Any]) -> str:
        return str(m.get("version") or m.get("created_at") or "")

    return max(models, key=key_fn)


def _artifact_path(model_type: str, version: str) -> Path:
    return MODEL_DIR / model_type / f"{version}.pkl"


# ------------------------------------------------------------
# Public API: Registering models
# ------------------------------------------------------------


def register_model(meta: Any) -> None:
    index = _load_registry()
    meta_dict = _meta_to_dict(meta)

    required = ["model_type", "version"]
    missing = [f for f in required if f not in meta_dict]
    if missing:
        raise ValueError(f"Metadata missing required fields: {missing}")

    meta_dict.setdefault("is_production", False)

    index.setdefault("models", []).append(meta_dict)
    _save_registry(index)

    logger.success(
        f"Registered model version={meta_dict['version']} type={meta_dict['model_type']}"
    )


# ------------------------------------------------------------
# Public API: Listing models
# ------------------------------------------------------------


def list_models(
    model_type: Optional[str] = None,
    production_only: bool = False,
) -> List[Dict[str, Any]]:
    index = _load_registry()
    return _filter_models(index, model_type=model_type, production_only=production_only)


# ------------------------------------------------------------
# Public API: Promote model
# ------------------------------------------------------------


def promote_model(model_type: str, version: str) -> None:
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
            f"No model found to promote: type={model_type}, version={version}"
        )

    _save_registry(index)
    logger.success(f"Promoted model type={model_type}, version={version} to production")


# ------------------------------------------------------------
# Public API: Delete model
# ------------------------------------------------------------


def delete_model(model_type: str, version: str) -> None:
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
            f"No model found to delete: type={model_type}, version={version}"
        )
    else:
        logger.info(f"Deleted registry entry for type={model_type}, version={version}")

    index["models"] = models
    _save_registry(index)

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
    index = _load_registry()
    candidates = _filter_models(
        index, model_type=model_type, production_only=production_only
    )

    if not candidates:
        raise ValueError(f"No models found for type={model_type}")

    if version is not None:
        candidates = [m for m in candidates if m.get("version") == version]
        if not candidates:
            raise ValueError(
                f"No models found for type={model_type}, version={version}"
            )
        meta = candidates[0]
    else:
        meta = _select_latest(candidates)

    model_version = meta["version"]
    path = _artifact_path(model_type, model_version)

    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path} for type={model_type}, version={model_version}"
        )

    raw_model = pd.read_pickle(path)
    model = ModelWrapper(raw_model)

    logger.info(f"Loaded model type={model_type}, version={model_version} from {path}")
    return model, meta
