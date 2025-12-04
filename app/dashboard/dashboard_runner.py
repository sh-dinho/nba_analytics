# ============================================================
# File: app/dashboard/dashboard_runner.py
# Purpose: Generate single daily dashboard for today's picks and bankroll
# ============================================================

import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
from scripts.simulate_bankroll import simulate_bankroll, plot_trajectory
from core.paths import RESULTS_DIR
from notifications import send_telegram_message, send_photo

logger = logging.getLogger("daily_dashboard")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main(input_csv: str, bankroll: float = 1000.0, strategy: str = "kelly", notify: bool = False):
    logger.info("🚀 Starting daily dashboard generation")

    today_str = datetime.today().strftime("%Y-%m-%d")
    output_dir = RESULTS_DIR / "dashboard"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load today's predictions
        df = pd.read_csv(input_csv)
        if df.empty:
            logger.warning("⚠️ No picks found for today.")
            return

        # Simulate bankroll for today's picks
        enriched, history, metrics = simulate_bankroll(
            preds_df=df,
            strategy=strategy,
            bankroll=bankroll
        )

        # Save enriched CSV
        daily_csv = output_dir / f"bankroll_{today_str}.csv"
        enriched.to_csv(daily_csv, index=False)
        logger.info(f"📑 Daily simulation saved to {daily_csv}")

        # Generate and save chart
        chart_path = output_dir / f"bankroll_{today_str}.png"
        if history:
            plot_trajectory(history, chart_path)

        # Telegram notification
        if notify:
            msg = (
                f"🏀 Daily Bankroll Dashboard ({today_str})\n"
                f"💰 Final Bankroll: {metrics['final_bankroll']:.2f}\n"
                f"📈 Win Rate: {metrics['win_rate']:.2%}\n"
                f"📊 Total Bets: {metrics['total_bets']}\n"
                f"💵 Avg EV: {metrics['avg_EV']:.3f}\n"
                f"🎯 Avg Kelly Bet: {metrics['avg_Kelly_Bet']:.2f}"
            )
            send_telegram_message(msg)
            if history:
                send_photo(str(chart_path), caption="📈 Daily Bankroll Trajectory")

        logger.info("✅ Daily dashboard generation complete")

    except Exception as e:
        logger.error(f"❌ Dashboard generation failed: {e}")
        raise

# === CLI ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate daily NBA picks dashboard")
    parser.add_argument("--input", type=str, required=True, help="CSV file with today's picks (prob + american_odds)")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Starting bankroll")
    parser.add_argument("--strategy", type=str, default="kelly", choices=["kelly", "flat"], help="Betting strategy")
    parser.add_argument("--notify", action="store_true", help="Send dashboard metrics and chart to Telegram")
    args = parser.parse_args()

    main(args.input, bankroll=args.bankroll, strategy=args.strategy, notify=args.notify)
