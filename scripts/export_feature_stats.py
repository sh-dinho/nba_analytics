# scripts/export_feature_stats.py
import logging
from config import configure_logging
from nba_analytics_core.db_module import export_feature_stats

if __name__ == "__main__":
    configure_logging()
    logging.info("Starting feature stats export...")
    try:
        export_feature_stats()
        logging.info("✔ Feature stats export complete")
    except Exception:
        logging.exception("❌ Feature stats export failed")