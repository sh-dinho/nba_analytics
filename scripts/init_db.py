# scripts/init_db.py
import logging
from config import configure_logging
from nba_analytics_core.db_module import init_db

if __name__ == "__main__":
    configure_logging()
    logging.info("Initializing DB...")
    try:
        init_db()
        logging.info("✔ DB initialized")
    except Exception:
        logging.exception("❌ DB initialization failed")