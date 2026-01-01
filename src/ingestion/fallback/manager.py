from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Fallback Manager
# File: src/ingestion/fallback/manager.py
# Author: Sadiq
#
# Description:
#     Coordinates multiple fallback sources. Applies them in
#     order until all missing team-game rows for a date are
#     filled or no more fallbacks apply.
# ============================================================

from datetime import date
from typing import List

import pandas as pd
from loguru import logger

from src.ingestion.fallback.base import FallbackSource


class FallbackManager:
    """
    Applies fallback sources in order.
    """

    def __init__(self, sources: List[FallbackSource]):
        self.sources = sources

    def fill_missing_for_date(self, day: date, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fallback sources sequentially until no missing rows remain.
        """
        out = df.copy()

        for source in self.sources:
            if source.can_fill(day, out):
                logger.info(f"[Fallback] Applying {source.__class__.__name__} for {day}")
                out = source.fill(day, out)

        return out