# scripts/run_daily_pipeline.py
from scripts.fetch_player_stats import main as fetch_stats
from scripts.generate_today_predictions import generate_today_predictions
from scripts.generate_picks import main as generate_picks
from scripts.utils import setup_logger

logger = setup_logger("daily_pipeline")

def run_pipeline():
    fetch_stats()
    preds = generate_today_predictions()
    picks = generate_picks()
    logger.info("Daily pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
