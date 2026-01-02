from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Canonical Backtest Engine
# File: src/backtest/canonical_engine.py
# Author: Sadiq
#
# Description:
#     Thin canonical wrapper around the Backtester.
#     No duplicated logic — delegates to engine.py.
# ============================================================

import pandas as pd

from src.backtest.config import BacktestConfig
from src.backtest.engine import Backtester, BacktestResult

# Alias for clarity in higher‑level code
CanonicalBacktestConfig = BacktestConfig


def run_canonical_backtest(
    predictions: pd.DataFrame,
    cfg: CanonicalBacktestConfig,
) -> BacktestResult:
    """
    Canonical entrypoint for running a backtest.
    """
    tester = Backtester(cfg)
    return tester.run(predictions)
