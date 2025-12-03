# ============================================================
# File: notifications.py
# Purpose: Send pipeline notifications (Telegram, summaries, charts)
# ============================================================

import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load Telegram credentials from environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram_message(msg: str) -> None:
    """
    Send a plain text message to Telegram.
    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials not set. Skipping notification.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
        if resp.status_code != 200:
            print(f"❌ Failed to send Telegram message: {resp.text}")
        else:
            print("✅ Telegram message sent successfully")
    except Exception as e:
        print(f"❌ Error sending Telegram message: {e}")


def send_photo(photo_path: str, caption: str = None) -> None:
    """Send a photo to Telegram with optional caption."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials not set. Skipping photo upload.")
        return
    if not os.path.exists(photo_path):
        print(f"⚠️ Chart not found at {photo_path}. Skipping photo upload.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as photo:
        files = {"photo": photo}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption or ""}
        try:
            resp = requests.post(url, data=data, files=files, timeout=10)
            if resp.status_code != 200:
                print(f"❌ Failed to send Telegram photo: {resp.text}")
            else:
                print("📸 Telegram chart sent successfully")
        except Exception as e:
            print(f"❌ Error sending Telegram photo: {e}")


def send_ev_summary(picks: pd.DataFrame) -> None:
    """
    Send an EV (expected value) summary of picks to Telegram.
    Expects a DataFrame with at least 'pick', 'ev', and 'stake_amount' columns.
    """
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

    msg = "EV Summary:\n" + "\n".join(summary_lines)
    send_telegram_message(msg)


def send_summary_report(summary_path: Path, chart_path: Path) -> None:
    """
    Send bankroll summary report (daily/weekly/monthly) to Telegram.
    Generates chart if missing.
    """
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

    # Generate chart if missing
    if not chart_path.exists() and "Final_Bankroll" in df.columns:
        plt.figure(figsize=(8, 5))
        x = df["Date"] if "Date" in df.columns else range(len(df))
        plt.plot(x, df["Final_Bankroll"], marker="o")
        plt.title("Bankroll Trajectory")
        plt.xlabel("Date")
        plt.ylabel("Final Bankroll")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(chart_path)
        print(f"📊 Chart generated at {chart_path}")

    send_photo(str(chart_path), caption="📈 Bankroll Trajectories")


def send_combined_dashboard(daily: Path, weekly: Path, monthly: Path, chart_path: Path) -> None:
    """
    Merge daily, weekly, and monthly summaries into one dashboard message.
    """
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
    msg = "*📊 Combined Dashboard*\n\n"
    for _, row in combined.iterrows():
        msg += (
            f"{row['Source']} → Bankroll={row.get('Final_Bankroll', 'N/A')}, "
            f"Avg EV={row.get('Avg_EV', 'N/A')}, Bets={row.get('Total_Bets', 'N/A')}\n"
        )

    send_telegram_message(msg)

    # Optional chart
    if "Final_Bankroll" in combined.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(combined.index, combined["Final_Bankroll"], marker="o")
        plt.title("Combined Bankroll Trajectories")
        plt.xlabel("Run Index")
        plt.ylabel("Final Bankroll")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(chart_path)
        print(f"📊 Combined chart generated at {chart_path}")
        send_photo(str(chart_path), caption="📈 Combined Bankroll Dashboard")