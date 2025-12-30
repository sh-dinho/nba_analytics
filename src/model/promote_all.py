from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Model Auto-Promotion Entrypoint
# File: src/model/promote_all.py
# Author: Sadiq
#
# Description:
#     Automatically promotes newly trained models to production
#     when they outperform the current production version,
#     based on model-type-specific metrics.
#
#     Works for:
#       - moneyline
#       - totals
#       - spread_regression
#       - spread_classification
#
#     Uses only public registry API:
#       - list_models(...)
#       - promote_model(...)
#
#     Run:
#         python -m src.model.promote_all
# ============================================================

from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from loguru import logger

from src.model.registry import list_models, promote_model


# ------------------------------------------------------------
# Promotion rules
# ------------------------------------------------------------


@dataclass
class PromotionRule:
    metric: str
    direction: str  # "lower" or "higher"


PROMOTION_RULES: Dict[str, List[PromotionRule]] = {
    # Prefer lower log_loss or higher accuracy
    "moneyline": [
        PromotionRule("test_log_loss", "lower"),
        PromotionRule("test_accuracy", "higher"),
    ],
    # Prefer lower RMSE
    "totals": [
        PromotionRule("test_rmse", "lower"),
    ],
    "spread_regression": [
        PromotionRule("test_rmse", "lower"),
    ],
    # Prefer lower log_loss or higher accuracy
    "spread_classification": [
        PromotionRule("test_log_loss", "lower"),
        PromotionRule("test_accuracy", "higher"),
    ],
}


# ------------------------------------------------------------
# Core helpers
# ------------------------------------------------------------


def _select_newest(models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Select latest model by version (timestamp string) or created_at as fallback.
    """
    if not models:
        raise ValueError("No models available to select newest from.")

    def key_fn(m: Dict[str, Any]) -> str:
        return str(m.get("version") or m.get("created_at") or "")

    return max(models, key=key_fn)


def _is_better(
    new_metrics: Dict[str, Any],
    prod_metrics: Dict[str, Any],
    rules: List[PromotionRule],
) -> bool:
    """
    Compare new model metrics vs production model metrics.
    Returns True if new model is better according to rules.
    """
    if not new_metrics or not prod_metrics:
        return False

    for rule in rules:
        if rule.metric not in new_metrics or rule.metric not in prod_metrics:
            continue

        new_val = new_metrics[rule.metric]
        prod_val = prod_metrics[rule.metric]

        if rule.direction == "lower" and new_val < prod_val:
            logger.info(
                f"Metric {rule.metric}: new={new_val:.4f} < prod={prod_val:.4f} "
                f"(better: lower)"
            )
            return True

        if rule.direction == "higher" and new_val > prod_val:
            logger.info(
                f"Metric {rule.metric}: new={new_val:.4f} > prod={prod_val:.4f} "
                f"(better: higher)"
            )
            return True

    return False


# ------------------------------------------------------------
# Auto-promotion per model type
# ------------------------------------------------------------


def auto_promote_model_type(model_type: str) -> Optional[str]:
    """
    Auto-promote the best model for a given model_type.
    Returns the promoted version or None if no promotion occurred.
    """
    logger.info(f"🔍 Auto-promotion check for model_type={model_type}")

    # All models of this type
    all_models = list_models(model_type=model_type, production_only=False)
    if not all_models:
        logger.warning(f"[AutoPromotion] No models found for type={model_type}")
        return None

    # Current production model (0 or 1)
    prod_models = list_models(model_type=model_type, production_only=True)
    prod = prod_models[0] if prod_models else None

    # Newest model
    newest = _select_newest(all_models)

    if prod and newest["version"] == prod["version"]:
        logger.info(
            f"[AutoPromotion] Newest model (version={newest['version']}) "
            f"is already production for type={model_type}"
        )
        return None

    rules = PROMOTION_RULES.get(model_type, [])
    if not rules:
        logger.error(
            f"[AutoPromotion] No promotion rules defined for type={model_type}"
        )
        return None

    newest_metrics = newest.get("metrics", {})
    prod_metrics = prod.get("metrics", {}) if prod else {}

    # If there is an existing production model, compare
    if prod:
        logger.info(
            f"[AutoPromotion] Comparing newest version={newest['version']} "
            f"vs production version={prod['version']} for type={model_type}"
        )

        if not _is_better(newest_metrics, prod_metrics, rules):
            logger.info(
                f"[AutoPromotion] New model version={newest['version']} "
                f"does NOT outperform production version={prod['version']} "
                f"for type={model_type} → no promotion."
            )
            return None

    # If no production model, we promote the newest by default
    logger.info(
        f"[AutoPromotion] Promoting version={newest['version']} as production "
        f"for type={model_type}"
    )
    promote_model(model_type=model_type, version=newest["version"])
    logger.success(
        f"🚀 Auto-promoted model_type={model_type}, version={newest['version']} "
        f"to production"
    )
    return newest["version"]


# ------------------------------------------------------------
# Unified auto-promotion for all model types
# ------------------------------------------------------------


def auto_promote_all() -> Dict[str, Optional[str]]:
    """
    Run auto-promotion for all v4 model types.

    Returns:
        Dict[model_type, promoted_version_or_None]
    """
    logger.info("🏁 Running auto-promotion for all v4 models")

    model_types = [
        "moneyline",
        "totals",
        "spread_regression",
        "spread_classification",
    ]

    promoted: Dict[str, Optional[str]] = {}

    for mt in model_types:
        try:
            promoted_version = auto_promote_model_type(mt)
            promoted[mt] = promoted_version
        except Exception as e:
            logger.error(f"[AutoPromotion] Failed for model_type={mt}: {e}")
            promoted[mt] = None

    logger.success("🏆 Auto-promotion run complete")
    return promoted


# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------

if __name__ == "__main__":
    results = auto_promote_all()
    for mt, ver in results.items():
        if ver:
            logger.success(f"[Summary] Promoted {mt} → version={ver}")
        else:
            logger.info(f"[Summary] No promotion for {mt}")
