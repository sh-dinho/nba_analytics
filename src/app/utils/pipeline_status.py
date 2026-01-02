from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Pipeline Status Utilities
# File: src/app/utils/pipeline_status.py
# Purpose:
#     Read heartbeat timestamps for ingestion, pipeline, and
#     other system components. Provides:
#       • safe timestamp parsing
#       • time-since-last-run helpers
#       • canonical heartbeat registry
# ============================================================

from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.config.paths import DATA_DIR


# ------------------------------------------------------------
# Heartbeat Registry
# ------------------------------------------------------------
HEARTBEATS = {
    "pipeline": DATA_DIR / "pipeline_last_run.txt",
    "ingestion": DATA_DIR / "ingestion_last_run.txt",
}


# ------------------------------------------------------------
# Timestamp Parsing
# ------------------------------------------------------------
def _parse_timestamp(raw: str) -> datetime | None:
    """
    Safely parse a timestamp string into a UTC datetime.
    Accepts:
        • ISO format
        • ISO with milliseconds
        • ISO with timezone offsets
        • strings ending with ' UTC'
    """
    raw = raw.strip().replace(" UTC", "")

    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None

    # Normalize to UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt


# ------------------------------------------------------------
# Read Heartbeat
# ------------------------------------------------------------
def read_timestamp(path: Path) -> datetime | None:
    """
    Read a heartbeat timestamp from disk.
    Returns:
        datetime (UTC) or None if missing/invalid.
    """
    if not path.exists():
        return None

    try:
        raw = path.read_text().strip()
        return _parse_timestamp(raw)
    except Exception:
        return None


# ------------------------------------------------------------
# Public Accessors
# ------------------------------------------------------------
def get_pipeline_last_run() -> datetime | None:
    return read_timestamp(HEARTBEATS["pipeline"])


def get_ingestion_last_run() -> datetime | None:
    return read_timestamp(HEARTBEATS["ingestion"])


# ------------------------------------------------------------
# Time Since Last Run
# ------------------------------------------------------------
def time_since(dt: datetime | None) -> str:
    """
    Human-readable time delta.
    Examples:
        "5 minutes ago"
        "2 hours ago"
        "No record"
    """
    if dt is None:
        return "No record"

    delta: timedelta = datetime.now(timezone.utc) - dt

    minutes = int(delta.total_seconds() // 60)
    hours = minutes // 60
    days = hours // 24

    if minutes < 1:
        return "Just now"
    if minutes < 60:
        return f"{minutes} minutes ago"
    if hours < 24:
        return f"{hours} hours ago"
    return f"{days} days ago"