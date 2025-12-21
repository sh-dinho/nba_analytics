from __future__ import annotations

from loguru import logger

from src.backtest.engine import Backtester, BacktestConfig
from src.alerts.telegram import send_telegram_message, send_bankroll_chart


def main():
    cfg = BacktestConfig(
        starting_bankroll=1000.0,
        min_edge=0.03,
        kelly_fraction=0.25,
        max_stake_fraction=0.05,
    )

    bt = Backtester(cfg)
    results = bt.run(start_date=None, end_date=None)

    if not results:
        logger.warning("Backtest produced no results.")
        return

    msg = (
        f"*Backtest Summary*\n"
        f"Final bankroll: `{results['final_bankroll']:.2f}`\n"
        f"Total profit: `{results['total_profit']:.2f}`\n"
        f"ROI: `{results['roi']:.3f}`\n"
        f"Hit rate: `{results['hit_rate']:.3f}`\n"
        f"Max drawdown: `{results['max_drawdown']:.3f}`\n"
    )
    send_telegram_message(msg)

    records = results["records"]
    if not records.empty:
        send_bankroll_chart(records, caption="Backtest Bankroll Curve")


if __name__ == "__main__":
    main()
