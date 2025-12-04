# ============================================================
# File: notifications.py
# Purpose: Send unified aggregation notifications (Telegram)
# ============================================================

import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from core.log_config import init_global_logger
from core.exceptions import PipelineError

logger = init_global_logger()

# ---------------- Telegram Credentials ----------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ---------------- Helper Functions ----------------
def _send_request(url: str, data: dict, files: dict | None = None, timeout: int = 10):
    """Send request to Telegram with error handling."""
    try:
        resp = requests.post(url, data=data, files=files, timeout=timeout)
        resp.raise_for_status()
        logger.info("✅ Telegram request succeeded")
        return resp
    except Exception as e:
        logger.error(f"❌ Telegram request failed: {e}")
        raise PipelineError(f"Telegram API request failed: {e}")


def _plot_chart(df: pd.DataFrame, chart_path: Path, title: str):
    """Generate and save bankroll chart."""
    plt.figure(figsize=(8, 5))
    x = df["Date"] if "Date" in df.columns else range(len(df))
    plt.plot(x, df["Avg_Final_Bankroll"], marker="o")
    plt.title(title)
    plt.xlabel("Date" if "Date" in df.columns else "Run Index")
    plt.ylabel("Avg Final Bankroll")
    plt.grid(True)
    plt.tight_layout()
    if chart_path.exists():
        chart_path.unlink()  # overwrite old chart
    plt.savefig(chart_path)
    plt.close()
    logger.info(f"📊 Chart saved to {chart_path}")


# ---------------- Notification Functions ----------------
def send_message(msg: str) -> None:
    """Send plain text message to Telegram (Markdown)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram credentials not set. Skipping message.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    _send_request(url, data)


def send_photo(photo_path: str, caption: str = None) -> None:
    """Send a photo to Telegram with optional caption."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram credentials not set. Skipping photo.")
        return
    if not Path(photo_path).exists():
        logger.warning(f"⚠️ Photo not found at {photo_path}. Skipping upload.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as photo:
        files = {"photo": photo}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption or ""}
        _send_request(url, data, files=files)


def send_dashboard(summary_path: Path, chart_path: Path, export_json: bool = False) -> None:
    """Send unified summary + chart to Telegram."""
    if not summary_path.exists():
        send_message(f"⚠️ Summary not found at {summary_path}")
        return

    df = pd.read_csv(summary_path)
    if df.empty:
        send_message("⚠️ Summary file is empty.")
        return

    # Generate chart if missing
    if not chart_path.exists():
        _plot_chart(df, chart_path, title="Unified Bankroll Trends")

    # Build summary message
    msg = "*📊 Unified Bankroll Summary*\n\n"
    for _, row in df.iterrows():
        msg += (
            f"🏷 Model: {row['Model']}\n"
            f"🏦 Avg Final Bankroll: {row['Avg_Final_Bankroll']:.2f}\n"
            f"✅ Win Rate: {row['Win_Rate']:.2%}\n"
            f"📊 Total Bets: {int(row['Total_Bets'])}\n\n"
        )

    # Highlight best model
    best_model = df.loc[df["Avg_Final_Bankroll"].idxmax()]
    msg += f"🚀 *Best Model:* {best_model['Model']} with bankroll {best_model['Avg_Final_Bankroll']:.2f}"

    send_message(msg)
    send_photo(str(chart_path), caption="📈 Unified Bankroll Chart")

    # Export JSON if requested
    if export_json:
        out_json = summary_path.with_suffix(".json")
        df.to_json(out_json, orient="records", indent=2)
        logger.info(f"📑 Summary also exported to {out_json}")
