"""
Public API exports for the model registry.

This module exposes the registry interface:
    - ModelMeta
    - load_registry
    - save_registry
    - register_model
    - load_production_model
    - promote_model
    - get_production_model_meta
"""

from .model_registry import (
    ModelMeta,
    load_registry,
    save_registry,
    register_model,
    load_production_model,
    promote_model,
    get_production_model_meta,
)

__all__: list[str] = [
    "ModelMeta",
    "load_registry",
    "save_registry",
    "register_model",
    "load_production_model",
    "promote_model",
    "get_production_model_meta",
]