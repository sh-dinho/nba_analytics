# Path: config.py
import os
import logging

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

def configure_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )