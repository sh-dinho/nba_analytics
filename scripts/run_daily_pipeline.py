# File: scripts/run_daily_pipeline.py

import logging
import datetime
import os
import pandas as pd
from app.prediction_pipeline import PredictionPipeline
import requests
import json

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
METRICS_LOG = f"{RESULTS_DIR}/pipeline_metrics.csv"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Optional Telegram notifications
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(text: str):
    """Send Telegram notification if credentials exist."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    except Exception as e:
        logging.warning(f"Telegram notification failed: {e}")

if __name__ == "__main__":
    try:
        pipeline = PredictionPipeline(
            threshold=0.6,
            strategy="kelly",
            max_fraction=0.05,
            use_synthetic=False # <-- NEW ARGUMENT ADDED HERE
        )

        df, metrics = pipeline.run()

        if metrics:
            logging.info("✅ Daily pipeline completed successfully")
            run_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            final_bankroll = metrics.get("final_bankroll_mean")
            roi = metrics.get("roi")

            # Safe fallback values
            fb_str = f"{final_bankroll:.2f}" if final_bankroll is not None else "0.00"
            roi_str = f"{roi * 100:.2f}%" if roi is not None else "0.00%"

            logging.info(f"Run timestamp: {run_time}")
            logging.info(f"Final Bankroll: {fb_str}")
            logging.info(f"ROI: {roi_str}")

            logging.info(
                f"📊 Pipeline summary: games={len(df)} | final_bankroll={fb_str} | roi={roi_str}"
            )

            # Append metrics to rolling CSV log
            log_entry = pd.DataFrame([{
                "timestamp": run_time,
                "games": len(df),
                "final_bankroll_mean": final_bankroll,
                "roi": roi
            }])

            log_entry.to_csv(
                METRICS_LOG,
                mode="a" if os.path.exists(METRICS_LOG) else "w",
                header=not os.path.exists(METRICS_LOG),
                index=False
            )

            logging.info(f"📁 Metrics appended to {METRICS_LOG}")

            send_telegram_message(f"✅ NBA Pipeline Success!\nGames: {len(df)}\nROI: {roi_str}")

        else:
            logging.warning("Pipeline completed but returned no metrics.")
            send_telegram_message("⚠️ NBA Pipeline completed but no metrics returned.")

    except Exception as e:
        logging.error(f"❌ Daily pipeline failed: {e}")
        send_telegram_message(f"❌ NBA Pipeline Failed!\nError: {e}")