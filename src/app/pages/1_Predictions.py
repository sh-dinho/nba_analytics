# ============================================================
# 🏀 NBA Analytics v4
# Module: Bet Tracker Engine
# Author: Sadiq
# Version: 4.0.0
# Purpose: Manages bet persistence with thread-safe file locks.
# ============================================================
import fcntl
from dataclasses import asdict

import pandas as pd

from app.engines.bet_tracker import BetRecord, _ensure_log_exists
from src.config.paths import BET_LOG_PATH


def append_bet(record: BetRecord):
    _ensure_log_exists()
    new_data = pd.DataFrame([asdict(record)])

    # FIXED: Added file locking for safe concurrent writing
    with open(BET_LOG_PATH, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            new_data.to_csv(f, header=False, index=False)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
