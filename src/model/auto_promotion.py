from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Model Auto-Promotion System
# File: src/model/auto_promotion.py
# Author: Sadiq
#
# Description:
#     Automatically promotes newly trained models to production
#     when they outperform the current production version.
#
#     Works for:
#       - moneyline
#       - totals
#       - spread_regression
#       - spread_classification
#
#     Promotion rules are metric-based and model-type specific.
# ============================================================

from dataclasses import dataclass
from typing import Dict, Any, Optional

from loguru import logger

from src.model.registry import (
    load_all_models_metadata,
    update_model_metadata,
)


# ------------------------------------------------------------
# Promotion rules
# ------------------------------------------------------------


@dataclass
class PromotionRule:
    metric: str
    direction: str  # "lower" or "higher"


PROMOTION_RULES = {
    "moneyline": [
        PromotionRule("test_log_loss", "lower"),
        PromotionRule("test_accuracy", "higher"),
    ],
    "totals": [
        PromotionRule("test_rmse", "lower"),
    ],
    "spread_regression": [
        PromotionRule("test_rmse", "lower"),
    ],
    "spread_classification": [
        PromotionRule("test_log_loss", "lower"),
        PromotionRule("test_accuracy", "higher"),
    ],
}


# ------------------------------------------------------------
# Core comparison logic
# ------------------------------------------------------------


def _is_better(
    new: Dict[str, Any], prod: Dict[str, Any], rules: list[PromotionRule]
) -> bool:
    """
    Compare new model metrics vs production model metrics.
    Returns True if new model is better according to rules.
    """
    for rule in rules:
        if rule.metric not in new or rule.metric not in prod:
            continue

        new_val = new[rule.metric]
        prod_val = prod[rule.metric]

        if rule.direction == "lower" and new_val < prod_val:
            return True
        if rule.direction == "higher" and new_val > prod_val:
            return True

    return False


# ------------------------------------------------------------
# Auto-promotion logic
# ------------------------------------------------------------


def auto_promote(model_type: str) -> Optional[str]:
    """
    Auto-promote the best model for a given model_type.
    Returns the promoted version or None.
    """
    logger.info(f"🔍 Auto-promotion check for model_type={model_type}")

    all_models = load_all_models_metadata(model_type)
    if not all_models:
        logger.warning(f"No models found for type={model_type}")
        return None

    # Identify production model
    prod = next((m for m in all_models if m["is_production"]), None)

    # Identify newest model (by version timestamp)
    newest = max(all_models, key=lambda m: m["version"])

    if prod and newest["version"] == prod["version"]:
        logger.info("Newest model is already production.")
        return None

    rules = PROMOTION_RULES.get(model_type, [])
    if not rules:
        logger.error(f"No promotion rules defined for model_type={model_type}")
        return None

    if prod:
        logger.info(
            f"Comparing newest={newest['version']} vs production={prod['version']}"
        )
        if not _is_better(newest["metrics"], prod["metrics"], rules):
            logger.info("New model does NOT outperform production → no promotion.")
            return None

    # Promote newest
    update_model_metadata(
        model_type=model_type,
        version=newest["version"],
        updates={"is_production": True},
    )

    # Demote old production
    if prod:
        update_model_metadata(
            model_type=model_type,
            version=prod["version"],
            updates={"is_production": False},
        )

    logger.success(f"🚀 Auto-promoted model {newest['version']} for type={model_type}")
    return newest["version"]


# ------------------------------------------------------------
# Run auto-promotion for all model types
# ------------------------------------------------------------


def auto_promote_all():
    logger.info("🏁 Running auto-promotion for all v4 models")

    model_types = [
        "moneyline",
        "totals",
        "spread_regression",
        "spread_classification",
    ]

    promoted = {}

    for mt in model_types:
        try:
            promoted_version = auto_promote(mt)
            promoted[mt] = promoted_version
        except Exception as e:
            logger.error(f"[AutoPromotion] Failed for {mt}: {e}")
            promoted[mt] = None

    logger.success("🏆 Auto-promotion complete")
    return promoted


if __name__ == "__main__":
    auto_promote_all()
