from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Train Model
# File: src/model/training/train_model.py
# Author: Sadiq
#
# Description:
#     CLI entrypoint for training a single model_type:
#         - moneyline
#         - totals
#         - spread
#
#     Uses:
#         - dataset_builder.build_dataset(model_type)
#         - training/<model_type>.py
#         - registry.save_model()
# ============================================================

import argparse
from loguru import logger

from src.model.training.dataset_builder import build_dataset
from src.model.training.moneyline import train_moneyline
from src.model.training.totals import train_totals
from src.model.training.spread import train_spread
from src.model.registry.save_model import save_model


# ------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------

TRAINERS = {
    "moneyline": train_moneyline,
    "totals": train_totals,
    "spread": train_spread,
}


def train_single_model(
    model_type: str,
    version: str,
    feature_version: str | None = None,
    model_family: str = "xgboost",
):
    logger.info(f"🚀 Training {model_type} model (version={version})")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test, feature_list = build_dataset(model_type)
    logger.debug(
        f"Dataset loaded: "
        f"X_train={X_train.shape}, X_test={X_test.shape}, "
        f"y_train={y_train.shape}, y_test={y_test.shape}"
    )

    # --------------------------------------------------------
    # Select trainer
    # --------------------------------------------------------
    if model_type not in TRAINERS:
        raise ValueError(f"Unsupported model_type: {model_type}")

    trainer = TRAINERS[model_type]

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------
    model, y_pred_or_prob, metrics = trainer(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        model_family=model_family,
    )

    logger.info(f"📊 Training metrics for {model_type}: {metrics}")

    # --------------------------------------------------------
    # Save + register model
    # --------------------------------------------------------
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


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["moneyline", "totals", "spread"],
        help="Which model to train",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Model version identifier (semantic, date-based, etc.)",
    )
    parser.add_argument(
        "--feature_version",
        type=str,
        default=None,
        help="Optional feature schema version",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        default="xgboost",
        choices=["xgboost", "lightgbm", "logistic_regression"],
        help="Underlying model family",
    )

    args = parser.parse_args()

    train_single_model(
        model_type=args.model_type,
        version=args.version,
        feature_version=args.feature_version,
        model_family=args.model_family,
    )


if __name__ == "__main__":
    main()