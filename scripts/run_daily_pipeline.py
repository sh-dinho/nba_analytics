# File: scripts/run_daily_pipeline.py

import logging
import datetime
from app.prediction_pipeline import PredictionPipeline

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
            logging.info(f"Run timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if "final_bankroll_mean" in metrics:
                logging.info(f"Final Bankroll: {metrics['final_bankroll_mean']:.2f}")
            if "roi" in metrics:
                logging.info(f"ROI: {metrics['roi']*100:.2f}%")
        else:
            logging.warning("Pipeline completed but no metrics were returned.")

    except Exception as e:
        logging.error(f"❌ Pipeline failed: {e}")