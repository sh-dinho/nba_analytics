# ============================================================
# File: app/prediction_pipeline.py
# Purpose: Run today's predictions using trained model and simulate bankroll/backtest
# ============================================================

import argparse
import os
import pandas as pd
from core.config import (
    MODEL_FILE_PKL,
    MODEL_FILE_H5,
    BANKROLL_FILE_TEMPLATE,
    PREDICTIONS_FILE,
    ensure_dirs,
)
from betting.bankroll import run_backtest  # deterministic backtest
import logging

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as tf_load_model
except ImportError:
    tf = None
    tf_load_model = None

ensure_dirs()

logger = logging.getLogger("prediction_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ============================================================
# Model Loader
# ============================================================

def load_model(model_type: str):
    """Load trained model depending on type."""
    if model_type in ("logistic", "xgb"):
        if not os.path.exists(MODEL_FILE_PKL):
            raise FileNotFoundError(f"{MODEL_FILE_PKL} not found")
        saved_obj = pd.read_pickle(MODEL_FILE_PKL) if MODEL_FILE_PKL.endswith(".pkl") else None
        return saved_obj.get("model") if saved_obj else None

    elif model_type == "nn":
        if not os.path.exists(MODEL_FILE_H5):
            raise FileNotFoundError(f"{MODEL_FILE_H5} not found")
        if tf is None:
            raise ImportError("TensorFlow is required to load NN model")
        return tf_load_model(MODEL_FILE_H5)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


# ============================================================
# Backtest / Prediction Wrapper
# ============================================================

def run_today_backtest(
    threshold: float = 0.6,
    strategy: str = "kelly",
    max_fraction: float = 0.05,
    model_type: str = "logistic",
):
    """
    Run today's prediction pipeline with deterministic bankroll backtest.
    Assumes predictions CSV already exists.
    """
    if not os.path.exists(PREDICTIONS_FILE):
        raise FileNotFoundError(f"{PREDICTIONS_FILE} not found. Run generate_today_predictions.py first.")

    df = pd.read_csv(PREDICTIONS_FILE)

    bankroll_df, metrics = run_backtest(
        df,
        initial=1000.0,
        strategy=strategy,
        max_fraction=max_fraction
    )

    bankroll_file = BANKROLL_FILE_TEMPLATE.format(model_type=model_type)
    bankroll_df.to_csv(bankroll_file, index=False)

    logger.info(f"✅ Bankroll trajectory saved → {bankroll_file}")
    logger.info(
        f"📊 Final Bankroll: ${metrics['final_bankroll']:.2f} | "
        f"ROI: {metrics['roi']:.2%} | "
        f"Wins: {metrics['wins']}/{metrics['total_bets']}"
    )

    return bankroll_df, metrics


# ============================================================
# CLI Entry
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run predictions + deterministic bankroll backtest.")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--strategy", type=str, default="kelly", choices=["kelly", "fixed"])
    parser.add_argument("--max_fraction", type=float, default=0.05)
    parser.add_argument(
        "--model_type",
        type=str,
        default="logistic",
        choices=["logistic", "xgb", "nn"]
    )
    args = parser.parse_args()

    run_today_backtest(
        threshold=args.threshold,
        strategy=args.strategy,
        max_fraction=args.max_fraction,
        model_type=args.model_type
    )


if __name__ == "__main__":
    main()
