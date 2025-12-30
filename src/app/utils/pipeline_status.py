from __future__ import annotations
from datetime import datetime
from pathlib import Path
from src.config.paths import DATA_DIR

PIPELINE_HEARTBEAT = DATA_DIR / "pipeline_last_run.txt"
INGESTION_HEARTBEAT = DATA_DIR / "ingestion_last_run.txt"


def read_timestamp(path: Path) -> str:
    if not path.exists():
        return "No record found"
    try:
        ts = path.read_text().strip()
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return "Invalid timestamp"


def get_pipeline_last_run() -> str:
    return read_timestamp(PIPELINE_HEARTBEAT)


def get_ingestion_last_run() -> str:
    return read_timestamp(INGESTION_HEARTBEAT)
