from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Model Registry
# File: src/model/registry/model_registry.py
# Author: Sadiq
#
# Description:
#     JSON-backed registry for storing model metadata,
#     loading production models, and promoting versions.
# ============================================================

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

import joblib
from loguru import logger

from src.config.paths import MODEL_REGISTRY_PATH, MODEL_DIR


# ------------------------------------------------------------
# Dataclass for model metadata
# ------------------------------------------------------------

@dataclass
class ModelMeta:
    model_type: str
    version: str
    model_name: str
    artifact_path: str

    # Optional metadata
    feature_version: Optional[str] = None
    train_start_date: Optional[str] = None
    train_end_date: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Registry metadata
    created_at_utc: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    is_production: bool = False
    promoted_at_utc: Optional[str] = None


# ------------------------------------------------------------
# Registry I/O
# ------------------------------------------------------------

def load_registry() -> Dict[str, Any]:
    if not MODEL_REGISTRY_PATH.exists():
        logger.warning(f"Model registry not found at {MODEL_REGISTRY_PATH}, initializing empty.")
        return {"models": []}

    try:
        return json.loads(MODEL_REGISTRY_PATH.read_text())
    except Exception as e:
        logger.error(f"Failed to load model registry: {e}")
        return {"models": []}


def save_registry(registry: Dict[str, Any]) -> None:
    MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
    logger.info(f"Model registry updated → {MODEL_REGISTRY_PATH}")


# ------------------------------------------------------------
# Register a new model
# ------------------------------------------------------------

def register_model(meta: ModelMeta) -> None:
    registry = load_registry()
    registry.setdefault("models", []).append(asdict(meta))
    save_registry(registry)
    logger.success(f"Registered model: {meta.model_name} ({meta.model_type}) v{meta.version}")


# ------------------------------------------------------------
# Load production model
# ------------------------------------------------------------

def get_production_model_meta(model_type: str) -> Optional[ModelMeta]:
    registry = load_registry()

    candidates = [
        m for m in registry.get("models", [])
        if m["model_type"] == model_type and m.get("is_production", False)
    ]

    if not candidates:
        return None

    # Sort by promotion timestamp, fallback to creation timestamp
    def sort_key(m):
        return m.get("promoted_at_utc") or m.get("created_at_utc")

    latest = sorted(candidates, key=sort_key)[-1]
    return ModelMeta(**latest)


def load_production_model(model_type: str):
    meta = get_production_model_meta(model_type)
    if meta is None:
        raise RuntimeError(f"No production model found for model_type={model_type}")

    artifact_path = MODEL_DIR / meta.artifact_path
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {artifact_path}")

    model = joblib.load(artifact_path)

    logger.info(
        f"Loaded production model: {meta.model_name} "
        f"({model_type}) v{meta.version} from {artifact_path}"
    )

    return model, meta


# ------------------------------------------------------------
# Promote a model version
# ------------------------------------------------------------

def promote_model(model_type: str, version: str) -> None:
    registry = load_registry()
    changed = False

    for m in registry.get("models", []):
        if m["model_type"] == model_type:
            if m["version"] == version:
                if not m.get("is_production", False):
                    m["is_production"] = True
                    m["promoted_at_utc"] = datetime.utcnow().isoformat()
                    changed = True
            else:
                if m.get("is_production", False):
                    m["is_production"] = False
                    changed = True

    if changed:
        save_registry(registry)
        logger.success(f"Promoted {model_type} model version {version} to production.")
    else:
        logger.info(f"No changes made during promotion for {model_type} v{version}.")