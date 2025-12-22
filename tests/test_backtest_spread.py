import pandas as pd

from src.backtest.engine_spread import SpreadBacktester, SpreadBacktestConfig


def test_spread_backtest_runs(monkeypatch):
    cfg = SpreadBacktestConfig()
    bt = SpreadBacktester(cfg)

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

    res = bt.run()
    assert res["final_bankroll"] > 1000
