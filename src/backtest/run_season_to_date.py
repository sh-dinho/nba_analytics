from __future__ import annotations

from loguru import logger

from src.backtest.engine import Backtester, BacktestConfig, current_season_date_range
from src.alerts.telegram import send_telegram_message, send_bankroll_chart


def main():
    start, end = current_season_date_range()
    logger.info(f"Running season-to-date backtest from {start} to {end}")

    cfg = BacktestConfig(
        starting_bankroll=1000.0,
        min_edge=0.03,
        kelly_fraction=0.25,
        max_stake_fraction=0.05,
    )

    bt = Backtester(cfg)
    results = bt.run(start_date=start, end_date=end)

    if not results:
        logger.warning("No results for season-to-date backtest.")
        return

    msg = (
        f"*Season-to-date Backtest*\n"
        f"Range: `{start}` → `{end}`\n"
        f"Final bankroll: `{results['final_bankroll']:.2f}`\n"
        f"Total profit: `{results['total_profit']:.2f}`\n"
        f"ROI: `{results['roi']:.3f}`\n"
        f"Hit rate: `{results['hit_rate']:.3f}`\n"
        f"Max drawdown: `{results['max_drawdown']:.3f}`\n"
        f"Bets: `{results['num_bets']}`, Wins: `{results['num_wins']}`, "
        f"Losses: `{results['num_losses']}`, Pushes: `{results['num_pushes']}`\n"
    )

    send_telegram_message(msg)

    records = results["records"]
    if not records.empty:
        send_bankroll_chart(records, caption="Season-to-date Bankroll Curve")


if __name__ == "__main__":
    main()
