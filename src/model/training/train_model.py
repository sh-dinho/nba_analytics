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
from __future__ import annotations

import argparse
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


def train_single_model(model_type: str, version: str, model_family: str = "xgboost") -> dict:
    if model_type not in TRAINERS:
        raise ValueError(f"Unknown model_type '{model_type}'")

    logger.info(f"🚀 Training {model_type} model (version={version}) using {model_family}")

    # Load dataset + metadata
    X_train, X_test, y_train, y_test, features, meta = build_dataset(model_type)

    logger.info(
        f"📦 Dataset ready: {len(X_train)} train rows, {len(X_test)} test rows, "
        f"{len(features)} features"
    )

    # Train
    trainer = TRAINERS[model_type]
    model, y_pred, report = trainer(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_family=model_family,
    )

    # Save model
    meta_obj = save_model(
        model=model,
        model_type=model_type,
        version=version,
        metrics=report,
        feature_list=features,
        model_family=model_family,
        train_start_date=meta["train_start_date"],
        train_end_date=meta["train_end_date"],
    )

    logger.success(f"🎉 Training complete: {meta_obj.model_name}")

    return {
        "ok": True,
        "model_type": model_type,
        "version": version,
        "model_name": meta_obj.model_name,
        "metrics": report,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def main():
    parser = argparse.ArgumentParser(description="Train a single NBA model")
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--family", default="xgboost")
    args = parser.parse_args()

    result = train_single_model(
        model_type=args.model_type,
        version=args.version,
        model_family=args.family,
    )

    logger.info(f"📊 Final metrics summary: {result['metrics']}")


if __name__ == "__main__":
    main()
