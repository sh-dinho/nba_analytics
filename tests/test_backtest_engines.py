import pandas as pd
from unittest.mock import MagicMock

from src.backtest.engine import Backtester, BacktestConfig
from src.backtest.engine_totals import TotalsBacktester, TotalsBacktestConfig
from src.backtest.engine_spread import SpreadBacktester, SpreadBacktestConfig


def test_backtest_ml(monkeypatch):
    monkeypatch.setattr(
        "src.backtest.engine.Backtester._load_joined_data",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "win_probability": [0.6],
                "price": [-110],
                "won": [1],
                "date": ["2024-01-01"],
            }
        ),
    )
    cfg = BacktestConfig(starting_bankroll=1000, min_edge=0, kelly_fraction=0.25)
    res = Backtester(cfg).run()
    assert res["final_bankroll"] > 1000


def test_backtest_totals(monkeypatch):
    monkeypatch.setattr(
        "src.backtest.engine_totals.TotalsBacktester._load_joined_data",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "predicted_total": [220],
                "market_total": [215],
                "actual_total": [225],
                "date": ["2024-01-01"],
            }
        ),
    )
    cfg = TotalsBacktestConfig()
    res = TotalsBacktester(cfg).run()
    assert res["final_bankroll"] > 1000


def test_backtest_spread(monkeypatch):
    monkeypatch.setattr(
        "src.backtest.engine_spread.SpreadBacktester._load_joined_data",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "predicted_margin": [5],
                "market_spread": [2],
                "actual_margin": [7],
                "date": ["2024-01-01"],
            }
        ),
    )
    cfg = SpreadBacktestConfig()
    res = SpreadBacktester(cfg).run()
    assert res["final_bankroll"] > 1000
