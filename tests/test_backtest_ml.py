import pandas as pd

from src.backtest.engine import Backtester, BacktestConfig


def test_ml_backtest_runs(monkeypatch):
    cfg = BacktestConfig(starting_bankroll=1000, min_edge=0, kelly_fraction=0.1)
    bt = Backtester(cfg)

    # Patch data loader
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

    res = bt.run()
    assert res["final_bankroll"] > 1000
