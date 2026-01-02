from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Model Training Orchestrator
# File: src/model/training.py
# Author: Sadiq
# ============================================================

from loguru import logger

from src.model.training.dataset_builder import build_dataset
from src.model.training.moneyline import train_moneyline
from src.model.training.totals import train_totals
from src.model.training.spread import train_spread
from src.model.registry.save_model import save_model
from src.model.training.hyperparams import MODEL_FAMILY_DEFAULTS


TRAINERS = {
    "moneyline": train_moneyline,
    "totals": train_totals,
    "spread": train_spread,
}


def train_model(
    model_type: str,
    version: str,
    model_family: str = "xgboost",
):
    """
    High-level training entrypoint for programmatic use.
    """

    logger.info(f"🚀 Training {model_type} model (version={version}) using {model_family}")

    # --------------------------------------------------------
    # 1. Build dataset
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test, feature_list, meta = build_dataset(model_type)

    # --------------------------------------------------------
    # 2. Select trainer
    # --------------------------------------------------------
    trainer = TRAINERS[model_type]

    # --------------------------------------------------------
    # 3. Train model
    # --------------------------------------------------------
    model, y_output, metrics = trainer(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_family=model_family,
    )

    # --------------------------------------------------------
    # 4. Save model + metadata
    # --------------------------------------------------------
    meta_obj = save_model(
        model=model,
        model_type=model_type,
        version=version,
        metrics=metrics,
        feature_list=feature_list,
        feature_version=meta["feature_version"],   # <-- REQUIRED FIX
        model_family=model_family,
        hyperparams=MODEL_FAMILY_DEFAULTS.get(model_family, {}),
        train_start_date=meta["train_start_date"],
        train_end_date=meta["train_end_date"],
    )

    logger.success(f"🎉 Training complete → {meta_obj.model_name}")

    return model, meta_obj