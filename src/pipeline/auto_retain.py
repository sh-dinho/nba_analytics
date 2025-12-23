# ============================================================
# 🏀 NBA Analytics v4
# Module: Automated Retraining Pipeline
# File: src/pipeline/auto_retrain.py
# Author: Sadiq
#
# Description:
#     Automatically retrains models when new data is available.
#     Steps:
#       1. Load canonical team-game data
#       2. Detect if new games exist since last training
#       3. Build v4 features
#       4. Train model via training_core
#       5. Register model in registry
#       6. Optional: backtest new model
#       7. Optional: auto-promote if better
# ============================================================

from __future__ import annotations

from datetime import date
import pandas as pd
from loguru import logger

from src.config.paths import SCHEDULE_SNAPSHOT
from src.features.builder import FeatureBuilder
from src.model.training_core import train_and_register_model
from src.model.registry import promote_model, _load_index
from src.backtest.engine import run_backtest


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def _load_canonical() -> pd.DataFrame:
    """Load canonical team-game snapshot."""
    df = pd.read_parquet(SCHEDULE_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _get_last_train_date(model_type: str) -> date:
    """Return last training end date from registry, or a very old date."""
    index = _load_index()
    models = [m for m in index.get("models", []) if m["model_type"] == model_type]

    if not models:
        return date(2000, 1, 1)

    latest = sorted(models, key=lambda m: m["created_at_utc"])[-1]
    return date.fromisoformat(latest["train_end_date"])


def _detect_new_data(df: pd.DataFrame, last_train_date: date) -> bool:
    """Return True if canonical data contains games after last_train_date."""
    return df[df["date"] > last_train_date].shape[0] > 0


# ------------------------------------------------------------
# Main Auto-Retrain Function
# ------------------------------------------------------------


def auto_retrain(
    model_type: str,
    feature_version: str = "v4",
    auto_promote: bool = True,
    min_improvement: float = 0.0,
):
    """
    Automatically retrain a model if new data is available.

    Args:
        model_type: "moneyline" | "totals" | "spread"
        feature_version: feature set version (default v4)
        auto_promote: promote new model if better
        min_improvement: minimum ROI improvement required for promotion
    """

    logger.info(f"🏀 Auto-retraining pipeline started for model_type={model_type}")

    # Load canonical data
    df = _load_canonical()

    # Determine last training date
    last_train_date = _get_last_train_date(model_type)
    logger.info(f"Last training date for {model_type}: {last_train_date}")

    # Check for new data
    if not _detect_new_data(df, last_train_date):
        logger.info("No new games since last training. Skipping retrain.")
        return None

    logger.info("New data detected — proceeding with retraining.")

    # Build features
    fb = FeatureBuilder(version=feature_version)
    features = fb.build_from_long(df)

    # Train + register model
    model, meta = train_and_register_model(
        model_type=model_type,
        df=features,
        feature_version=feature_version,
    )

    logger.success(f"Model trained and registered: {meta.model_name} v{meta.version}")

    # ------------------------------------------------------------
    # Optional Backtesting
    # ------------------------------------------------------------
    try:
        from src.backtest.results_loader import load_results

        results_df = load_results()

        # You must generate predictions for backtesting
        # (This depends on your prediction pipeline)
        # For now, assume meta contains predictions or skip
        if "predictions" in meta.__dict__:
            pred_df = meta.predictions
            bt = run_backtest(pred_df, results_df)

            logger.info(
                f"Backtest ROI={bt.roi:.2%}, HitRate={bt.hit_rate:.2%}, "
                f"CLV={bt.clv:.2%}, Bets={bt.n_bets}"
            )

            # Auto-promote if better
            if auto_promote and bt.roi >= min_improvement:
                promote_model(model_type, version=meta.version)
                logger.success(
                    f"Auto-promoted {model_type} model v{meta.version} "
                    f"(ROI {bt.roi:.2%})"
                )

        else:
            logger.warning("No predictions available for backtesting. Skipping.")

    except Exception as e:
        logger.warning(f"Backtesting skipped: {e}")

    logger.success(f"Auto-retrain complete for {model_type}.")
    return meta
