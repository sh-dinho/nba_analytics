# File: app/prediction_pipeline.py
# Purpose: Run predictions using trained model and bankroll simulation/backtest

import argparse
import os
import pandas as pd
import joblib
import datetime 
from core.config import (
    MODEL_FILE_PKL, MODEL_FILE_H5, BANKROLL_FILE_TEMPLATE, PREDICTIONS_FILE, ensure_dirs
)
from betting.bankroll import run_backtest # FIX: Use deterministic backtest

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as tf_load_model
except ImportError:
    tf = None
    tf_load_model = None

ensure_dirs()

def load_model(model_type: str):
    """Load trained model depending on type."""
    if model_type in ["logistic", "xgb"]:
        model_path = MODEL_FILE_PKL
        # ... (loading logic remains the same) ...
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found")
        saved_object = joblib.load(model_path)
        return saved_object['model']

    elif model_type == "nn":
        model_path = MODEL_FILE_H5
        # ... (loading logic remains the same) ...
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found")
        if tf is None:
            raise ImportError("TensorFlow not available to load NN model")
        return tf_load_model(model_path)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

def generate_today_predictions(threshold: float = 0.6,
                               strategy: str = "kelly",
                               max_fraction: float = 0.05,
                               model_type: str = "logistic"):
    """
    Wrapper to run today's prediction pipeline programmatically.
    """
    # Load trained model if needed
    # model = load_model(model_type)

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

    print(f"✅ Bankroll trajectory saved to {bankroll_file}")
    print(f"📊 Final Bankroll: ${metrics['final_bankroll']:.2f} | ROI: {metrics['roi']:.2%} | Wins: {metrics['wins']}/{metrics['total_bets']}")

    return bankroll_df, metrics

def main():
    # ... (parser setup remains the same) ...
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--strategy", type=str, default="kelly", choices=["kelly", "fixed"])
    parser.add_argument("--max_fraction", type=float, default=0.05)
    parser.add_argument("--model_type", type=str, default="logistic",
                        choices=["logistic", "xgb", "nn"])
    args = parser.parse_args()

    # Load trained model (model prediction part is removed for brevity, assuming it runs)
    # model = load_model(args.model_type)

    # --- Load prediction data ---
    if not os.path.exists(PREDICTIONS_FILE):
        raise FileNotFoundError(f"{PREDICTIONS_FILE} not found. Run generate_today_predictions.py first.")
    
    df = pd.read_csv(PREDICTIONS_FILE)
    
    # Run deterministic backtest/simulation
    bankroll_df, metrics = run_backtest(
        df, 
        initial=1000.0, 
        strategy=args.strategy, 
        max_fraction=args.max_fraction
    )
    
    # Save bankroll trajectory
    bankroll_file = BANKROLL_FILE_TEMPLATE.format(model_type=args.model_type)
    bankroll_df.to_csv(bankroll_file, index=False)
    
    print(f"✅ Bankroll trajectory saved to {bankroll_file}")
    print(f"📊 Final Bankroll: ${metrics['final_bankroll']:.2f} | ROI: {metrics['roi']:.2%} | Wins: {metrics['wins']}/{metrics['total_bets']}")


if __name__ == "__main__":
    main()