import os
import logging
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
DB_PATH = os.environ.get("DB_PATH", "data/nba.db")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

def configure_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )