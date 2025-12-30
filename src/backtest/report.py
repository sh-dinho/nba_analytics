# File: src/backtesting/report.py
from backtest.engine import BacktestResult


def backtest_report(result: BacktestResult):
    print("=== Backtest Summary ===")
    print(f"ROI: {result.roi:.2%}")
    print(f"Hit Rate: {result.hit_rate:.2%}")
    print(f"Avg Edge: {result.avg_edge:.2%}")
    print(f"CLV: {result.clv:.2%}")
    print(f"Bets: {result.n_bets}")
