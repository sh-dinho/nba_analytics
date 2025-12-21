# ============================================================
# Project: NBA Analytics & Betting Engine
# Module: Backtest & Accuracy Reporting
# Author: Sadiq
#
# Description:
#     Generate client-ready HTML reports that summarize:
#       - Backtest performance (ROI, bankroll, drawdown, win/loss)
#       - Model accuracy (overall + per-season)
#       - Executive Summary (auto-generated insights)
# ============================================================

from __future__ import annotations

from datetime import datetime
from pathlib import Path

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


def _generate_executive_summary(bt, acc):
    if not bt or acc.total_examples == 0:
        return ["Insufficient data to generate a meaningful summary."]

    insights = []

    if bt["roi"] > 0.05:
        insights.append(
            f"Strategy achieved a strong ROI of {bt['roi']:.1%}, indicating consistent profitability."
        )
    elif bt["roi"] > 0:
        insights.append(f"Strategy produced a modest positive ROI of {bt['roi']:.1%}.")
    else:
        insights.append(
            f"Strategy underperformed with an ROI of {bt['roi']:.1%}, suggesting parameter tuning is needed."
        )

    if bt["hit_rate"] > 0.55:
        insights.append(
            f"Hit rate of {bt['hit_rate']:.1%} exceeds typical market baselines."
        )
    else:
        insights.append(
            f"Hit rate of {bt['hit_rate']:.1%} is within expected variance for NBA moneyline models."
        )

    if bt["max_drawdown"] > -0.20:
        insights.append(
            "Drawdown remained well-controlled, indicating stable risk exposure."
        )
    else:
        insights.append(
            "Drawdown exceeded 20%, suggesting the strategy may be too aggressive."
        )

    if acc.overall_accuracy > 0.60:
        insights.append(
            f"Model accuracy of {acc.overall_accuracy:.1%} is strong for NBA forecasting."
        )
    else:
        insights.append(
            f"Model accuracy of {acc.overall_accuracy:.1%} is typical for NBA forecasting models."
        )

    if bt["num_bets"] > 200:
        insights.append(
            f"High bet volume ({bt['num_bets']}) provides strong statistical confidence."
        )
    else:
        insights.append(
            f"Lower bet volume ({bt['num_bets']}) suggests results should be interpreted cautiously."
        )

    return insights[:5]


def _render_html_report(start_date, end_date, config, bt, acc) -> str:
    def fmt(x, digits=3):
        return f"{x:.{digits}f}"

    date_range_str = f"{start_date or 'start'} → {end_date or 'end'}"

    if not bt:
        bt_html = "<p>No backtest results available.</p>"
        summary_items = ["No backtest results available."]
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
        summary_items = _generate_executive_summary(bt, acc)

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

    summary_html = (
        "<ul>" + "".join(f"<li>{item}</li>" for item in summary_items) + "</ul>"
    )

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

        <h2>Executive Summary</h2>
        {summary_html}

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
