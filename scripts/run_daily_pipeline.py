# File: scripts/run_daily_pipeline.py

import logging
import datetime
import os
import pandas as pd
from app.prediction_pipeline import PredictionPipeline

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
METRICS_LOG = f"{RESULTS_DIR}/pipeline_metrics.csv"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

if __name__ == "__main__":
    try:
        pipeline = PredictionPipeline(threshold=0.6, strategy="kelly", max_fraction=0.05)
        df, metrics = pipeline.run()

        if metrics:
            logging.info("✅ Daily pipeline completed successfully")
            run_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logging.info(f"Run timestamp: {run_time}")

            final_bankroll = metrics.get("final_bankroll_mean", None)
            roi = metrics.get("roi", None)

            if final_bankroll is not None:
                logging.info(f"Final Bankroll: {final_bankroll:.2f}")
            if roi is not None:
                logging.info(f"ROI: {roi*100:.2f}%")

            # Summary line
            logging.info(
                f"📊 Pipeline summary: games={len(df)} | "
                f"final_bankroll={final_bankroll:.2f if final_bankroll else 0} | "
                f"roi={roi*100:.2f if roi else 0}%"
            )

            # Append metrics to rolling CSV log
            log_entry = pd.DataFrame([{
                "timestamp": run_time,
                "games": len(df),
                "final_bankroll_mean": final_bankroll,
                "roi": roi
            }])

            if os.path.exists(METRICS_LOG):
                log_entry.to_csv(METRICS_LOG, mode="a", header=False, index=False)
            else:
                log_entry.to_csv(METRICS_LOG, index=False)

            logging.info(f"📁 Metrics appended to {METRICS_LOG}")

        else:
            logging.warning("Pipeline completed but no metrics were returned.")

    except Exception as e:
        logging.error(f"❌ Pipeline failed: {e}")