# scripts/run_daily_pipeline.py
import os
import logging
import pandas as pd
from app.predict_pipeline import generate_predictions, build_bets_from_predictions
from scripts.simulate_bankroll import simulate_bankroll
from core.utils import send_telegram_message

def main(threshold: float = 0.6, strategy: str = "kelly", max_fraction: float = 0.05):
    logging.basicConfig(level=logging.INFO)
    preds = generate_predictions(threshold=threshold)
    if preds.empty:
        logging.info("No predictions today.")
        return

    bets = build_bets_from_predictions(preds, threshold=threshold)
    if bets.empty:
        logging.info("No qualifying bets.")
        return

    sim = simulate_bankroll(bets, strategy=strategy, max_fraction=max_fraction)

    try:
        send_telegram_message(f"Pipeline complete: {len(bets)} bets generated. Latest bankroll: {sim.iloc[-1]['bankroll']:.2f}")
    except Exception:
        pass

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    preds.to_csv("artifacts/predictions_today.csv", index=False)
    bets.to_csv("artifacts/bets_today.csv", index=False)
    sim.to_csv("artifacts/simulation_today.csv", index=False)
    logging.info("✔ Artifacts saved.")

if __name__ == "__main__":
    main()