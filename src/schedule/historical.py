# ============================================================
# File: src/run_pipeline.py
# Purpose: Run the NBA daily pipeline end-to-end
# ============================================================

import logging
from pathlib import Path
import pandas as pd

# -------------------------
# Local module imports
# -------------------------
from src.config.pipeline_config import Config
from src.schedule.historical import download_historical_schedule
from src.schedule.compare import compare_schedules
from src.features.engineer import prepare_schedule
from src.prediction.model import load_model, predict_games
from src.prediction.rank import generate_rankings

# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------
# Paths & config
# -------------------------
config = Config()
RAW_PATH = Path(config.paths["raw"])
CACHE_PATH = Path(config.paths["cache"])
HISTORY_PATH = Path(config.paths["history"])
MODEL_PATH = Path(config.paths["model"])


def main():
    logger.info("===== NBA DAILY PIPELINE START =====")

    # -------------------------
    # 1️⃣ Download historical schedule
    # -------------------------
    historical_schedule = download_historical_schedule(config)
    if historical_schedule.empty:
        logger.error("No historical schedule available. Exiting.")
        return
    logger.info(f"Historical schedule loaded (rows={len(historical_schedule)})")

    # -------------------------
    # 2️⃣ Load incremental/master schedule
    # -------------------------
    incremental_file = CACHE_PATH / "master_schedule.parquet"
    if incremental_file.exists():
        master_schedule = pd.read_parquet(incremental_file)
    else:
        master_schedule = historical_schedule.copy()

    # -------------------------
    # 3️⃣ Generate today schedule (optional)
    # -------------------------
    today_schedule = pd.DataFrame()  # Could fetch live data here
    if today_schedule.empty:
        logger.warning("Today schedule is empty; skipping comparison.")

    # -------------------------
    # 4️⃣ Compare and update
    # -------------------------
    if not today_schedule.empty:
        changes = compare_schedules(today_schedule, historical_schedule)
        if not changes.empty:
            master_schedule = pd.concat([master_schedule, changes], ignore_index=True)
            logger.info(f"New games added: {len(changes)}")
    else:
        logger.info("No new games today.")

    # -------------------------
    # 5️⃣ Save incremental schedule
    # -------------------------
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    master_schedule.to_parquet(incremental_file, index=False)
    logger.info(f"Incremental schedule updated (rows={len(master_schedule)})")

    # -------------------------
    # 6️⃣ Feature engineering
    # -------------------------
    features = prepare_schedule(master_schedule)
    logger.info("Features prepared for ML model")

    # -------------------------
    # 7️⃣ Load model & predict
    # -------------------------
    try:
        model = load_model(MODEL_PATH)
        predictions = predict_games(features, model)
        logger.info("Predictions completed")
    except FileNotFoundError:
        predictions = pd.DataFrame()
        logger.warning("Model not found. Skipping predictions.")

    # -------------------------
    # 8️⃣ Generate rankings & betting
    # -------------------------
    if not predictions.empty:
        rankings = generate_rankings(predictions)
        logger.info("Rankings generated successfully")
    else:
        logger.warning("Skipping rankings & betting due to missing predictions")

    logger.info("===== NBA DAILY PIPELINE END =====")


if __name__ == "__main__":
    main()
