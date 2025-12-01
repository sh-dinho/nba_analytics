# scripts/run_daily_pipeline.py
import logging
from config import configure_logging
from nba_analytics_core.db_module import init_db
from nba_analytics_core.predictor import predict_todays_games
from nba_analytics_core.simulate_ai_bankroll import simulate_ai_strategy

def main():
    configure_logging()
    logging.info("Starting daily pipeline...")
    try:
        init_db()
        preds = predict_todays_games()
        logging.info(f"✔ Predictions generated: {len(preds)} rows")
        sim = simulate_ai_strategy(initial_bankroll=1000, strategy="kelly")
        logging.info(f"✔ Simulation completed: ROI={sim.kpis['roi']:.3f}, WinRate={sim.kpis['win_rate']:.3f}")
    except Exception:
        logging.exception("❌ Daily pipeline failed")

if __name__ == "__main__":
    main()