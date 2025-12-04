# ============================================================
# File: scripts/run_daily_pipeline_cli.py
# Purpose: CLI entrypoint for daily NBA analytics pipeline
# ============================================================

import argparse
from pathlib import Path

# --- Config constants ---
from core.config import (
    DEFAULT_BANKROLL,
    MAX_KELLY_FRACTION,
    EV_THRESHOLD,
    MIN_KELLY_STAKE,
    PRINT_ONLY_ACTIONABLE,
)

# --- Path constants ---
from core.paths import (
    DATA_DIR,
    LOGS_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    ARCHIVE_DIR,
    SUMMARY_FILE,
    PICKS_FILE,
    PICKS_BANKROLL_FILE,
    FEATURES_FILE,
    FEATURES_LOG_FILE,
    CONFIG_LOG_FILE,
)

# --- Notifications ---
from notifications import send_telegram_message, send_photo

# --- Pipeline runner (replace with your actual runner function) ---
from scripts.xgb_runner import xgb_runner


def main():
    parser = argparse.ArgumentParser(description="Run daily NBA analytics pipeline")
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL,
                        help="Initial bankroll for simulation")
    parser.add_argument("--kelly", action="store_true",
                        help="Use Kelly criterion for bet sizing")
    parser.add_argument("--notify", action="store_true",
                        help="Send results to Telegram")
    args = parser.parse_args()

    # Example: load data (replace with your actual data loading logic)
    # For now, just placeholders
    data = []  # feature matrix
    todays_games_uo = []
    frame_ml = None
    games = []
    home_odds = []
    away_odds = []

    # Run pipeline
    try:
        results, history, metrics = xgb_runner(
            data,
            todays_games_uo,
            frame_ml,
            games,
            home_odds,
            away_odds,
            use_kelly=args.kelly,
            bankroll=args.bankroll,
            max_fraction=MAX_KELLY_FRACTION,
        )
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return

    # Print summary
    print("=== Daily Pipeline Summary ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Optional Telegram notification
    if args.notify:
        msg = (
            f"🏀 Daily Pipeline Results\n"
            f"Final Bankroll: {metrics['final_bankroll']}\n"
            f"Avg EV: {metrics['avg_EV']}\n"
            f"Win Rate: {metrics['win_rate']}\n"
            f"Total Bets: {metrics['total_bets']}\n"
            f"Actionable Games: {metrics['actionable_count']}"
        )
        send_telegram_message(msg)
        chart_path = RESULTS_DIR / "daily_bankroll.png"
        # If you generate a chart elsewhere, send it too
        if chart_path.exists():
            send_photo(str(chart_path), caption="📈 Daily Bankroll Trajectory")


if __name__ == "__main__":
    main()