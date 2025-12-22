import pandas as pd

from src.backtest.engine_totals import TotalsBacktester, TotalsBacktestConfig


def test_totals_backtest_runs(monkeypatch):
    cfg = TotalsBacktestConfig()
    bt = TotalsBacktester(cfg)

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

    res = bt.run()
    assert res["final_bankroll"] > 1000
