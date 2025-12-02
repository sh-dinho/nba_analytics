# ============================================================
# File: app/prediction_pipeline.py
# Purpose: Run predictions using trained model and bankroll simulation
# ============================================================

import argparse
import os
import pandas as pd
import joblib
from scripts.Utils import Simulation

try:
    import tensorflow as tf
except ImportError:
    tf = None


def load_model(model_type: str):
    """
    Load trained model depending on type.
    """
    if model_type in ["logistic", "xgb"]:
        model_path = "models/game_predictor.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found")
        return joblib.load(model_path)

    elif model_type == "nn":
        model_path = "models/game_predictor.h5"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found")
        if tf is None:
            raise ImportError("TensorFlow not available to load NN model")
        return tf.keras.models.load_model(model_path)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--strategy", type=str, default="kelly")
    parser.add_argument("--max_fraction", type=float, default=0.05)
    parser.add_argument("--model_type", type=str, default="logistic",
                        choices=["logistic", "xgb", "nn"])
    args = parser.parse_args()

    # Load trained model
    model = load_model(args.model_type)

    # Example synthetic data (replace with real features)
    X = pd.DataFrame({
        "feature1": [0.1, 0.2, 0.8, 0.9],
        "feature2": [1, 2, 3, 4]
    })

    # Predict probabilities
    if args.model_type == "nn":
        probs = model.predict(X).flatten()
    else:
        probs = model.predict_proba(X)[:, 1]

    # Build bets list
    bets = []
    for prob in probs:
        bets.append({"prob_win": prob, "odds": 2.0})  # placeholder odds

    # Run bankroll simulation
    sim = Simulation(initial_bankroll=1000)
    sim.run(bets, strategy=args.strategy, max_fraction=args.max_fraction)

    # Save results
    df = pd.DataFrame(sim.history)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/picks_bankroll.csv", index=False)

    # Print summary
    print("📊 Pipeline Summary:", sim.summary())


if __name__ == "__main__":
    main()