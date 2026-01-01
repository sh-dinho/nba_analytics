from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Auto Retrain
# File: src/pipeline/auto_retrain.py
# Author: Sadiq
#
# Description:
#     Automatically retrains models when new data arrives.
#     Steps:
#         1. Load latest feature snapshot
#         2. Build features
#         3. Train new models
#         4. Evaluate against production models
#         5. Promote if better
#
#     Safe for cron, Airflow, GitHub Actions, or systemd timers.
# ============================================================

import pandas as pd
from loguru import logger

from src.features.builder import FeatureBuilder
from src.model.training.dataset_builder import build_dataset
from src.model.training.common import train_model_common
from src.model.training.metrics import compute_metrics
from src.model.registry import load_production_model, promote_model
from src.model.registry.save_model import save_model


DEFAULT_MODEL_TYPES = ["moneyline", "totals", "spread"]


def auto_retrain(
    feature_version: str,
    model_version: str,
    model_types: list[str] = None,
) -> dict:
    """
    Run the full auto‑retrain pipeline and return a structured result.

    Parameters
    ----------
    feature_version : str
        Version tag for feature builder.
    model_version : str
        Version tag for saved models.
    model_types : list[str]
        Which model types to retrain (default: moneyline, totals, spread).

    Returns
    -------
    dict
        Structured result with per‑model metrics and promotion decisions.
    """

    logger.info("🔄 Starting auto‑retrain pipeline...")

    if model_types is None:
        model_types = DEFAULT_MODEL_TYPES

    results: dict[str, dict] = {}

    # --------------------------------------------------------
    # 1. Load long-format snapshot
    # --------------------------------------------------------
    try:
        fb = FeatureBuilder(version=feature_version)
        df_long = fb.load_raw_snapshot()
    except Exception as e:
        msg = f"Failed to load long snapshot: {e}"
        logger.error(msg)
        return {"ok": False, "error": msg}

    if df_long.empty:
        msg = "Long snapshot is empty — cannot retrain."
        logger.error(msg)
        return {"ok": False, "error": msg}

    logger.info(f"Loaded long snapshot: {len(df_long)} rows")

    # --------------------------------------------------------
    # 2. Build features
    # --------------------------------------------------------
    try:
        features = fb.build(df_long)
    except Exception as e:
        msg = f"Feature building failed: {e}"
        logger.error(msg)
        return {"ok": False, "error": msg}

    if features.empty:
        msg = "Feature builder returned empty DataFrame."
        logger.error(msg)
        return {"ok": False, "error": msg}

    logger.info(f"Built features: {features.shape}")

    # --------------------------------------------------------
    # 3. Train + evaluate each model type
    # --------------------------------------------------------
    for model_type in model_types:
        logger.info(f"🚀 Retraining {model_type} model...")

        # Build dataset
        try:
            X_train, X_test, y_train, y_test, feature_list = build_dataset(model_type)
        except Exception as e:
            logger.error(f"Dataset build failed for {model_type}: {e}")
            results[model_type] = {"ok": False, "error": str(e)}
            continue

        # Train new model
        try:
            new_model, y_pred_or_prob = train_model_common(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                model_family="xgboost",
            )
        except Exception as e:
            logger.error(f"Training failed for {model_type}: {e}")
            results[model_type] = {"ok": False, "error": str(e)}
            continue

        # Compute new metrics
        new_metrics = compute_metrics(model_type, y_test, y_pred_or_prob)
        logger.info(f"📊 New {model_type} metrics: {new_metrics}")

        # ----------------------------------------------------
        # Load production model for comparison
        # ----------------------------------------------------
        try:
            prod_model, prod_meta = load_production_model(model_type)
            logger.info(f"Loaded production model: {prod_meta.model_name}")

            if model_type == "moneyline":
                prod_pred = prod_model.predict_proba(X_test)[:, 1]
            else:
                prod_pred = prod_model.predict(X_test)

            prod_metrics = compute_metrics(model_type, y_test, prod_pred)
            logger.info(f"📊 Production {model_type} metrics: {prod_metrics}")

        except Exception:
            logger.warning(f"No production model found for {model_type}. Will promote new model.")
            prod_metrics = None

        # ----------------------------------------------------
        # Decide whether to promote
        # ----------------------------------------------------
        if prod_metrics is None:
            should_promote = True
        else:
            metric_key = "log_loss" if model_type == "moneyline" else "rmse"
            should_promote = new_metrics[metric_key] < prod_metrics[metric_key]

        # ----------------------------------------------------
        # Save new model
        # ----------------------------------------------------
        try:
            meta = save_model(
                model=new_model,
                model_type=model_type,
                version=model_version,
                feature_version=feature_version,
                metrics=new_metrics,
                train_start_date=str(X_train.index.min()),
                train_end_date=str(X_train.index.max()),
            )
        except Exception as e:
            logger.error(f"Failed to save {model_type} model: {e}")
            results[model_type] = {"ok": False, "error": str(e)}
            continue

        # ----------------------------------------------------
        # Promote if better
        # ----------------------------------------------------
        if should_promote:
            promote_model(model_type, model_version)
            logger.success(f"🎉 Promoted new {model_type} model → {meta.model_name}")
        else:
            logger.info(f"⚖️ New {model_type} model NOT promoted")

        results[model_type] = {
            "ok": True,
            "new_metrics": new_metrics,
            "prod_metrics": prod_metrics,
            "promoted": should_promote,
            "model_name": meta.model_name,
        }

    logger.success("✨ Auto‑retrain pipeline complete.")
    return {"ok": True, "results": results}