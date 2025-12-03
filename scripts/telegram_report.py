# ============================================================
# File: scripts/telegram_report.py
# Purpose: Send bankroll summary + chart + trend analysis to Telegram
# ============================================================

import os
import requests
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from core.log_config import setup_logger
from core.exceptions import PipelineError, DataError

logger = setup_logger("telegram_report")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

REQUIRED_COLS = {"Model", "Final_Bankroll", "Win_Rate", "Avg_EV", "Avg_Stake", "Total_Bets"}


def send_message(text: str):
    """Send a text message to Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram credentials not set. Skipping report.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("📲 Telegram text report sent successfully")
    except Exception as e:
        logger.error(f"❌ Failed to send Telegram text report: {e}")
        raise PipelineError(f"Telegram message failed: {e}")


def send_photo(photo_path: str, caption: str = None):
    """Send a photo to Telegram with optional caption."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram credentials not set. Skipping photo upload.")
        return
    if not os.path.exists(photo_path):
        logger.warning(f"⚠️ Chart not found at {photo_path}. Skipping photo upload.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as photo:
        files = {"photo": photo}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption or ""}
        try:
            resp = requests.post(url, data=data, files=files, timeout=10)
            resp.raise_for_status()
            logger.info("📸 Telegram chart sent successfully")
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram chart: {e}")
            raise PipelineError(f"Telegram photo failed: {e}")


def load_summary(summary_path: str) -> pd.DataFrame:
    """Load and validate the summary CSV."""
    if not os.path.exists(summary_path):
        logger.warning(f"⚠️ No summary file found at {summary_path}. Skipping report.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        raise DataError(f"Failed to read {summary_path}: {e}")

    if df.empty:
        logger.warning("⚠️ Summary file is empty. Skipping report.")
        return pd.DataFrame()

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise DataError(f"Missing expected columns in summary: {missing}")

    return df


def generate_chart(df: pd.DataFrame, chart_path: str):
    """Generate bankroll trajectory chart if not already present."""
    if "Final_Bankroll" in df.columns:
        plt.figure(figsize=(8, 5))
        if "Date" in df.columns:
            x = df["Date"]
        elif "timestamp" in df.columns:
            x = df["timestamp"]
        else:
            x = range(len(df))
        plt.plot(x, df["Final_Bankroll"], marker="o")
        plt.title("Bankroll Trajectory")
        plt.xlabel("Date")
        plt.ylabel("Final Bankroll")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(chart_path)
        logger.info(f"📊 Chart generated at {chart_path}")


def format_message(df: pd.DataFrame) -> str:
    """Format the summary DataFrame into a Telegram message."""
    message = "*🏀 NBA Bankroll Report*\n\n"
    for _, row in df.iterrows():
        message += (
            f"📌 Model: {row['Model']}\n"
            f"🏦 Final Bankroll: {row['Final_Bankroll']:.2f}\n"
            f"✅ Win Rate: {row['Win_Rate']:.2%}\n"
            f"💰 Avg EV: {row['Avg_EV']:.2f}\n"
            f"🎯 Avg Stake: {row['Avg_Stake']:.2f}\n"
            f"📊 Total Bets: {int(row['Total_Bets'])}\n\n"
        )

    # Trend analysis
    try:
        best_model = df.loc[df["Final_Bankroll"].fillna(0).idxmax()]
        message += (
            f"📈 *Trend Analysis:* {best_model['Model']} leads with bankroll {best_model['Final_Bankroll']:.2f}."
        )
        if len(df) > 1:
            delta = df["Final_Bankroll"].iloc[-1] - df["Final_Bankroll"].iloc[-2]
            message += f"\n📉 Change since last run: {delta:+.2f}"
    except Exception as e:
        logger.warning(f"⚠️ Trend analysis failed: {e}")

    # Truncate if too long for Telegram
    if len(message) > 4000:
        message = message[:4000] + "\n... (truncated)"

    return message


def main(summary_path: str, chart_path: str):
    """Load summary, format report, and send to Telegram."""
    df = load_summary(summary_path)
    if df.empty:
        return

    # Generate chart if missing
    if not os.path.exists(chart_path):
        generate_chart(df, chart_path)

    message = format_message(df)
    send_message(message)
    send_photo(chart_path, caption="📈 Bankroll Trajectories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send bankroll summary report to Telegram")
    parser.add_argument("--summary", default="results/summary.csv",
                        help="Path to summary CSV (daily, weekly, or monthly)")
    parser.add_argument("--chart", default="results/bankroll.png",
                        help="Path to bankroll chart image")
    args = parser.parse_args()

    main(summary_path=args.summary, chart_path=args.chart)