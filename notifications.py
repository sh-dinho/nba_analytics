# ============================================================
# File: notifications.py
# Purpose: Send pipeline notifications (Telegram, summaries, charts)
# ============================================================

import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from core.log_config import init_global_logger
from core.exceptions import PipelineError

logger = init_global_logger()

# Load Telegram credentials from environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def _send_request(url: str, data: dict, files: dict | None = None, timeout: int = 10):
    """Helper to send requests with timeout and error handling."""
    try:
        resp = requests.post(url, data=data, files=files, timeout=timeout)
        resp.raise_for_status()
        logger.info("✅ Telegram request succeeded")
        return resp
    except Exception as e:
        logger.error(f"❌ Telegram request failed: {e}")
        raise PipelineError(f"Telegram API request failed: {e}")

def _plot_chart(df: pd.DataFrame, chart_path: Path, title: str, y_col: str = "Final_Bankroll"):
    """Centralized chart plotting helper."""
    plt.figure(figsize=(8, 5))
    x = df["Date"] if "Date" in df.columns else range(len(df))
    plt.plot(x, df[y_col], marker="o")
    plt.title(title)
    plt.xlabel("Date" if "Date" in df.columns else "Run Index")
    plt.ylabel(y_col)
    plt.grid(True)
    plt.tight_layout()
    if chart_path.exists():
        chart_path.unlink()  # overwrite old chart
    plt.savefig(chart_path)
    logger.info(f"📊 Chart generated at {chart_path}")

# ----------------------------------------------------------------
# Notification functions
# ----------------------------------------------------------------
def send_telegram_message(msg: str) -> None:
    """Send a plain text message to Telegram (Markdown enabled)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram credentials not set. Skipping notification.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    _send_request(url, data)

def send_photo(photo_path: str, caption: str = None) -> None:
    """Send a photo to Telegram with optional caption."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram credentials not set. Skipping photo upload.")
        return
    if not os.path.exists(photo_path):
        logger.warning(f"⚠️ Chart not found at {photo_path}. Skipping photo upload.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as photo:
        files = {"photo": photo}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption or ""}
        _send_request(url, data, files=files)

def send_ev_summary(picks: pd.DataFrame) -> None:
    """Send an EV (expected value) summary of picks to Telegram."""
    if picks is None or picks.empty:
        send_telegram_message("No picks available for EV summary.")
        return
    summary_lines = []
    if "pick" in picks.columns and "ev" in picks.columns:
        grouped = picks.groupby("pick").agg(
            avg_ev=("ev", "mean"),
            total_stake=("stake_amount", "sum") if "stake_amount" in picks.columns else ("ev", "count")
        ).reset_index()
        for _, row in grouped.iterrows():
            summary_lines.append(
                f"{row['pick']}: avg EV={row['avg_ev']:.3f}, total stake={row['total_stake']:.2f}"
            )
    else:
        summary_lines.append("⚠️ Picks DataFrame missing 'pick' or 'ev' columns.")
    msg = "*EV Summary:*\n" + "\n".join(summary_lines)
    send_telegram_message(msg)

def send_summary_report(summary_path: Path, chart_path: Path, export_json: bool = False) -> None:
    """Send bankroll summary report (daily/weekly/monthly) to Telegram."""
    if not summary_path.exists():
        send_telegram_message(f"⚠️ No summary file found at {summary_path}")
        return
    df = pd.read_csv(summary_path)
    if df.empty:
        send_telegram_message("⚠️ Summary file is empty.")
        return
    # Format message
    msg = f"*🏀 Bankroll Summary ({summary_path.name})*\n\n"
    for _, row in df.iterrows():
        msg += (
            f"🏦 Final Bankroll: {row.get('Final_Bankroll', 'N/A')}\n"
            f"✅ Win Rate: {row.get('Win_Rate', 'N/A')}\n"
            f"💰 Avg EV: {row.get('Avg_EV', 'N/A')}\n"
            f"🎯 Avg Stake: {row.get('Avg_Stake', 'N/A')}\n"
            f"📊 Total Bets: {row.get('Total_Bets', 'N/A')}\n\n"
        )
    send_telegram_message(msg)
    # Export JSON if requested
    if export_json:
        out_json = summary_path.with_suffix(".json")
        df.to_json(out_json, orient="records", indent=2)
        logger.info(f"📑 Summary also exported to {out_json}")
    # Generate chart if missing
    if not chart_path.exists() and "Final_Bankroll" in df.columns:
        _plot_chart(df, chart_path, "Bankroll Trajectory")
    send_photo(str(chart_path), caption="📈 Bankroll Trajectories")

def send_combined_dashboard(daily: Path, weekly: Path, monthly: Path, chart_path: Path, export_json: bool = False) -> None:
    """Merge daily, weekly, and monthly summaries into one dashboard message."""
    dfs = []
    for p in [daily, weekly, monthly]:
        if p.exists():
            df = pd.read_csv(p)
            if not df.empty:
                df["Source"] = p.stem
                dfs.append(df)
    if not dfs:
        send_telegram_message("⚠️ No summaries available for dashboard.")
        return
    combined = pd.concat(dfs, ignore_index=True)
    if "Date" in combined.columns:
        combined = combined.sort_values(by="Date")
    msg = "*📊 Combined Dashboard*\n\n"
    for _, row in combined.iterrows():
        msg += (
            f"{row['Source']} → Bankroll={row.get('Final_Bankroll', 'N/A')}, "
            f"Avg EV={row.get('Avg_EV', 'N/A')}, Bets={row.get('Total_Bets', 'N/A')}\n"
        )
    send_telegram_message(msg)
    if export_json:
        out_json = chart_path.with_suffix(".json")
        combined.to_json(out_json, orient="records", indent=2)
        logger.info(f"📑 Combined dashboard also exported to {out_json}")
    if "Final_Bankroll" in combined.columns:
        _plot_chart(combined, chart_path, "Combined Bankroll Trajectories")
        send_photo(str(chart_path), caption="📈 Combined Bankroll Dashboard")

# ----------------------------------------------------------------
# Aliases for compatibility
# ----------------------------------------------------------------
send_message = send_telegram_message