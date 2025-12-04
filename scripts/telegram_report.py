# ============================================================
# File: scripts/telegram_report.py
# Purpose: Send bankroll summary + chart + trend analysis to Telegram
# ============================================================

import os
import requests
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from core.log_config import init_global_logger
from core.exceptions import PipelineError, DataError

logger = init_global_logger()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

REQUIRED_COLS = {"Model", "Final_Bankroll", "Win_Rate", "Avg_EV", "Avg_Stake", "Total_Bets"}

SUMMARY_MAP = {
    "daily": Path("results/summary.csv"),
    "weekly": Path("results/bankroll_simulation_weekly.csv"),
    "monthly": Path("results/bankroll_simulation_monthly.csv"),
}

CHART_MAP = {
    "daily": Path("results/bankroll.png"),
    "weekly": Path("results/weekly_bankroll_chart.png"),
    "monthly": Path("results/monthly_bankroll_chart.png"),
}


# === Telegram Helpers ===

def send_message(text: str):
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


# === Data Helpers ===

def load_summary(summary_path: Path) -> pd.DataFrame:
    if not summary_path.exists():
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


def generate_chart(df: pd.DataFrame, chart_path: Path):
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


def format_message(df: pd.DataFrame, summary_type: str) -> str:
    message = f"*🏀 NBA Bankroll Report ({summary_type.capitalize()})*\n\n"
    for _, row in df.iterrows():
        message += (
            f"📌 Model: {row['Model']}\n"
            f"🏦 Final Bankroll: {row['Final_Bankroll']:.2f}\n"
            f"✅ Win Rate: {row['Win_Rate']:.2%}\n"
            f"💰 Avg EV: {row['Avg_EV']:.2f}\n"
            f"🎯 Avg Stake: {row['Avg_Stake']:.2f}\n"
            f"📊 Total Bets: {int(row['Total_Bets'])}\n\n"
        )
    try:
        best_model = df.loc[df["Final_Bankroll"].fillna(0).idxmax()]
        message += f"📈 *Trend Analysis:* {best_model['Model']} leads with bankroll {best_model['Final_Bankroll']:.2f}."
        if len(df) > 1:
            delta = df["Final_Bankroll"].iloc[-1] - df["Final_Bankroll"].iloc[-2]
            message += f"\n📉 Change since last run: {delta:+.2f}"
        if "Cumulative_Bankroll" in df.columns:
            last_val = df["Cumulative_Bankroll"].iloc[-1]
            prev_val = df["Cumulative_Bankroll"].iloc[-2] if len(df) > 1 else last_val
            delta_cum = last_val - prev_val
            message += f"\n💹 Cumulative bankroll progression: {last_val:.2f} ({delta_cum:+.2f} since last period)"
    except Exception as e:
        logger.warning(f"⚠️ Trend analysis failed: {e}")
    if len(message) > 4000:
        message = message[:4000] + "\n... (truncated)"
    return message


def send_report(summary_type: str, export_json: bool = False):
    summary_path = SUMMARY_MAP[summary_type]
    chart_path = CHART_MAP[summary_type]
    df = load_summary(summary_path)
    if df.empty:
        return
    if not chart_path.exists():
        generate_chart(df, chart_path)
    message = format_message(df, summary_type)
    send_message(message)
    send_photo(str(chart_path), caption=f"📈 {summary_type.capitalize()} Bankroll Trajectories")
    if export_json:
        out_json = summary_path.with_suffix(".json")
        try:
            df.to_json(out_json, orient="records", indent=2)
            logger.info(f"📑 Summary also exported to {out_json}")
        except Exception as e:
            logger.warning(f"Failed to export summary to JSON: {e}")


def main(summary_type: str, export_json: bool = False, all_reports: bool = False):
    if all_reports:
        for stype in ["daily", "weekly", "monthly"]:
            send_report(stype, export_json=export_json)
    else:
        send_report(summary_type, export_json=export_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send bankroll summary report to Telegram")
    parser.add_argument("--summary-type", choices=["daily", "weekly", "monthly"], default="daily",
                        help="Which summary to send (daily, weekly, monthly)")
    parser.add_argument("--export-json", action="store_true",
                        help="Also export summary to JSON format")
    parser.add_argument("--all", action="store_true",
                        help="Send all three reports (daily, weekly, monthly)")
    args = parser.parse_args()

    try:
        main(summary_type=args.summary_type, export_json=args.export_json, all_reports=args.all)
    except Exception as e:
        logger.error(f"❌ Telegram report failed: {e}")
        raise PipelineError(f"Telegram report failed: {e}")