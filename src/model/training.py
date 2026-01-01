from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Model Training Orchestrator
# File: src/model/training.py
# Author: Sadiq
#
# Description:
#     High-level programmatic entrypoint for training a model.
#     Delegates to:
#         - dataset_builder
#         - training wrappers
#         - save_model
# ============================================================

from loguru import logger

from src.model.training.dataset_builder import build_dataset
from src.model.training.moneyline import train_moneyline
from src.model.training.totals import train_totals
from src.model.training.spread import train_spread
from src.model.registry.save_model import save_model


TRAINERS = {
    "moneyline": train_moneyline,
    "totals": train_totals,
    "spread": train_spread,
}


def train_model(
    model_type: str,
    version: str,
    feature_version: str | None = None,
    model_family: str = "xgboost",
):
    """
    High-level training entrypoint for programmatic use.
    """

    logger.info(f"🚀 Training {model_type} model (version={version})")

    # Build dataset
    X_train, X_test, y_train, y_test, feature_list = build_dataset(model_type)

    # Select trainer
    trainer = TRAINERS[model_type]

    # Train model
    model, y_output, metrics = trainer(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        model_family=model_family,
    )

    # Save model
    meta = save_model(
        model=model,
        model_type=model_type,
        version=version,
        feature_version=feature_version,
        metrics=metrics,
        train_start_date=str(X_train.index.min()),
        train_end_date=str(X_train.index.max()),
    )

    logger.success(f"🎉 Training complete → {meta.model_name}")

    return model, meta