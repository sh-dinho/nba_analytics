from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Fallback Base Class
# File: src/ingestion/fallback/base.py
# Author: Sadiq
#
# Description:
#     Abstract base class for ingestion fallback providers.
#     Each fallback must implement:
#         - can_fill(day, df)
#         - fill(day, df)
# ============================================================

from abc import ABC, abstractmethod
from datetime import date
import pandas as pd


class FallbackSource(ABC):
    """
    Base class for fallback providers.
    """

    @abstractmethod
    def can_fill(self, day: date, df: pd.DataFrame) -> bool:
        """
        Return True if this fallback source can fill missing rows
        for the given date.
        """
        raise NotImplementedError

    @abstractmethod
    def fill(self, day: date, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame with missing rows filled.
        """
        raise NotImplementedError