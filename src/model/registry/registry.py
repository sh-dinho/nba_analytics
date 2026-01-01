from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Model Registry
# File: src/model/registry.py
# Author: Sadiq
#
# Description:
#     JSON-backed model registry for storing and retrieving
#     model metadata and locating production models.
# ============================================================

from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional, List

from loguru import logger

from src.config.paths import MODEL_REGISTRY_PATH, MODEL_DIR


# ------------------------------------------------------------
# Dataclass
# ------------------------------------------------------------

@dataclass
class ModelMeta:
    model_type: str               # "moneyline" | "totals" | "spread"
    version: str                  # e.g. "1.0.0" or "2025-01-01"
    model_name: str               # human-readable name
    artifact_path: str            # relative path under MODEL_DIR
    feature_version: Optional[str] = None
    train_start_date: Optional[str] = None
    train_end_date: Optional[str] = None
    created_at_utc: str = datetime.utcnow().isoformat()
    metrics: Dict[str, Any] = None
    is_production: bool = False


# ------------------------------------------------------------
# Registry I/O
# ------------------------------------------------------------

def load_registry() -> Dict[str, Any]:
    if not MODEL_REGISTRY_PATH.exists():
        logger.warning(f"Model registry not found at {MODEL_REGISTRY_PATH}, initializing empty.")
        return {"models": []}

    try:
        with MODEL_REGISTRY_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load model registry: {e}")
        return {"models": []}


def save_registry(registry: Dict[str, Any]) -> None:
    MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    logger.info(f"Model registry updated → {MODEL_REGISTRY_PATH}")


# ------------------------------------------------------------
# Registration
# ------------------------------------------------------------

def register_model(meta: ModelMeta) -> None:
    registry = load_registry()
    models: List[Dict[str, Any]] = registry.get("models", [])

    models.append(asdict(meta))
    registry["models"] = models

    save_registry(registry)
    logger.success(f"Registered model: {meta.model_name} ({meta.model_type}) v{meta.version}")


# ------------------------------------------------------------
# Production model lookup
# ------------------------------------------------------------

def get_production_model_meta(model_type: str) -> Optional[ModelMeta]:
    registry = load_registry()

    candidates = [
        m for m in registry.get("models", [])
        if m["model_type"] == model_type and m.get("is_production", False)
    ]

    if not candidates:
        return None

    latest = sorted(candidates, key=lambda m: m["created_at_utc"])[-1]
    return ModelMeta(**latest)


def load_production_model(model_type: str):
    meta = get_production_model_meta(model_type)
    if meta is None:
        raise RuntimeError(f"No production model found for model_type={model_type}")

    artifact_path = MODEL_DIR / meta.artifact_path
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {artifact_path}")

    import joblib
    model = joblib.load(artifact_path)

    logger.info(
        f"Loaded production model: {meta.model_name} "
        f"({model_type}) v{meta.version} from {artifact_path}"
    )

    return model, meta


# ------------------------------------------------------------
# Promotion
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