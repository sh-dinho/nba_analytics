# ============================================================
# Project: NBA Analytics & Betting Engine
# Module: Backtest & Accuracy Reporting
# Author: Sadiq
#
# Description:
#     Generate client-ready HTML reports that summarize:
#       - Backtest performance (ROI, bankroll, drawdown, win/loss)
#       - Model accuracy (overall + per-season)
#
# ============================================================

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pandas as pd

from src.backtest.engine import Backtester, BacktestConfig
from src.backtest.accuracy import AccuracyEngine
from src.config.paths import DATA_DIR


REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_backtest_accuracy_report(
    start_date: str | None,
    end_date: str | None,
    config: BacktestConfig,
    decision_threshold: float = 0.5,
) -> Path:
    bt = Backtester(config)
    bt_results = bt.run(start_date=start_date, end_date=end_date)

    acc_engine = AccuracyEngine(threshold=decision_threshold)
    acc_results = acc_engine.run(start_date=start_date, end_date=end_date)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = (
        REPORTS_DIR
        / f"report_{start_date or 'all'}_{end_date or 'all'}_{timestamp}.html"
    )

    html = _render_html_report(
        start_date=start_date,
        end_date=end_date,
        config=config,
        bt=bt_results,
        acc=acc_results,
    )
    report_path.write_text(html, encoding="utf-8")
    return report_path


def _render_html_report(start_date, end_date, config, bt, acc) -> str:
    def fmt(x, digits=3):
        return f"{x:.{digits}f}"

    date_range_str = f"{start_date or 'start'} → {end_date or 'end'}"

    if not bt:
        bt_html = "<p>No backtest results available.</p>"
    else:
        bt_html = f"""
        <h2>Backtest Summary</h2>
        <ul>
          <li>Starting bankroll: {fmt(config.starting_bankroll, 2)}</li>
          <li>Final bankroll: {fmt(bt['final_bankroll'], 2)}</li>
          <li>Total profit: {fmt(bt['total_profit'], 2)}</li>
          <li>ROI: {fmt(bt['roi'])}</li>
          <li>Hit rate: {fmt(bt['hit_rate'])}</li>
          <li>Max drawdown: {fmt(bt['max_drawdown'])}</li>
          <li>Bets: {bt['num_bets']}, Wins: {bt['num_wins']}, Losses: {bt['num_losses']}, Pushes: {bt['num_pushes']}</li>
        </ul>
        """

    if acc.total_examples == 0:
        acc_html = "<p>No accuracy data available.</p>"
    else:
        by_season_table = acc.by_season.to_html(
            index=False, float_format=lambda x: f"{x:.3f}"
        )
        acc_html = f"""
        <h2>Model Accuracy</h2>
        <ul>
          <li>Overall accuracy: {fmt(acc.overall_accuracy)}</li>
          <li>Total examples: {acc.total_examples}</li>
        </ul>
        <h3>Accuracy by Season</h3>
        {by_season_table}
        """

    html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>NBA Analytics Report</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 2rem; }}
          h1, h2, h3 {{ color: #222; }}
          table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
          th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: right; }}
          th {{ background-color: #f5f5f5; }}
          td:first-child, th:first-child {{ text-align: left; }}
        </style>
      </head>
      <body>
        <h1>NBA Analytics & Betting Engine — Performance Report</h1>
        <p><strong>Date range:</strong> {date_range_str}</p>

        <h2>Strategy Configuration</h2>
        <ul>
          <li>Starting bankroll: {fmt(config.starting_bankroll, 2)}</li>
          <li>Minimum edge: {fmt(config.min_edge)}</li>
          <li>Kelly fraction: {fmt(config.kelly_fraction)}</li>
          <li>Max stake fraction: {fmt(config.max_stake_fraction)}</li>
        </ul>

        {bt_html}

        {acc_html}
      </body>
    </html>
    """
    return html
