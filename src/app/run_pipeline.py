from datetime import datetime
from src.config.paths import DATA_DIR

PIPELINE_HEARTBEAT = DATA_DIR / "pipeline_last_run.txt"
INGESTION_HEARTBEAT = DATA_DIR / "ingestion_last_run.txt"


def write_pipeline_heartbeat():
    PIPELINE_HEARTBEAT.write_text(datetime.utcnow().isoformat())


def write_ingestion_heartbeat():
    INGESTION_HEARTBEAT.write_text(datetime.utcnow().isoformat())
