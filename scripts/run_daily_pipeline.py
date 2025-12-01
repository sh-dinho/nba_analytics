# scripts/run_daily_pipeline.py
import os
import logging
import pandas as pd
from app.predict_pipeline import generate_predictions, build_bets_from_predictions
from scripts.simulate_bankroll import simulate_bankroll
from nba_core.utils import send_telegram_message

def main(threshold: float = 0.6, strategy: str = "kelly", max_fraction: float = 0.05):
    logging.basicConfig(level=logging.INFO)
    preds = generate_predictions(threshold=threshold)
    if preds.empty:
        logging.info("No predictions today.")
        return

    bets = build_bets_from_predictions(preds, threshold=threshold)
    if bets.empty:
        logging.info("No qualifying bets (based on probability AND positive EV).")
        return

    sim = simulate_bankroll(bets, strategy=strategy, max_fraction=max_fraction)

    # Calculate Max Drawdown for risk assessment
    # Max Drawdown = max(Peak - Trough) / Peak
    bankroll_history = sim['bankroll']
    cumulative_max = bankroll_history.cummax()
    drawdown = cumulative_max - bankroll_history
    max_drawdown_amount = drawdown.max()
    max_drawdown_percentage = (drawdown / cumulative_max).max()

    try:
        # Include Max Drawdown in the notification
        send_telegram_message(
            f"Pipeline complete: {len(bets)} value bets generated. "
            f"Latest bankroll: ${bankroll_history.iloc[-1]:.2f}. "
            f"Max Drawdown: {max_drawdown_percentage:.2%}" # NEW: Max Drawdown included
        )
        # 
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