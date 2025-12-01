# scripts/update_db.py
import logging
from datetime import datetime
from config import configure_logging
from nba_analytics_core.db_module import load_season_to_db, export_feature_stats

def update_db(start_season: int = 2023) -> None:
    configure_logging()
    current_year = datetime.now().year
    logging.info(f"Starting DB update for seasons {start_season}–{current_year}")
    for season in range(start_season, current_year + 1):
        try:
            load_season_to_db(season)
            logging.info(f"✔ Loaded season {season}")
        except Exception as e:
            logging.exception(f"❌ Failed to load season {season}: {e}")
    export_feature_stats()
    logging.info("✔ Feature statistics refreshed after DB update")

if __name__ == "__main__":
    update_db()