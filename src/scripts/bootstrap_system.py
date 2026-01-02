from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Script: Full System Bootstrap
# Purpose:
#   Initialize the entire analytics system from scratch:
#       • Repair historical data (2022 → today)
#       • Build canonical snapshots
#       • Train all models
#       • Register models in the registry
#       • Run predictions for today
#       • Validate outputs
#       • Write heartbeats
# ============================================================

from datetime import date
from loguru import logger

# --- 1. Historical Repair -----------------------------------
from src.scripts.repair_data import total_repair

# --- 2. Model Training ---------------------------------------
from src.model.training.train_moneyline import train_moneyline_model
from src.model.training.train_totals import train_totals_model
from src.model.training.train_spread import train_spread_model

# --- 3. Model Registry ---------------------------------------
from src.model.registry.model_registry import register_model

# --- 4. Prediction Orchestrator ------------------------------
from src.model.prediction.run_predictions import run_prediction_for_date

# --- 5. Heartbeats -------------------------------------------
from src.config.paths import DATA_DIR
from pathlib import Path
from datetime import datetime


PIPELINE_HEARTBEAT = DATA_DIR / "pipeline_last_run.txt"
BOOTSTRAP_HEARTBEAT = DATA_DIR / "bootstrap_last_run.txt"


def _utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def _write_heartbeat(path: Path, label: str) -> None:
    ts = _utc_timestamp()
    try:
        path.write_text(ts)
        logger.info(f"[Bootstrap] Wrote {label} heartbeat → {ts}")
    except Exception as e:
        logger.error(f"[Bootstrap] Failed to write {label} heartbeat: {e}")


# ------------------------------------------------------------
# Main Bootstrap Routine
# ------------------------------------------------------------
def bootstrap_system() -> None:
    logger.info("====================================================")
    logger.info("🚀 Starting Full System Bootstrap (v5 Canonical)")
    logger.info("====================================================")

    # --------------------------------------------------------
    # 1. Repair Historical Data
    # --------------------------------------------------------
    logger.info("🛠️ Step 1 — Repairing historical data...")
    total_repair()

    # --------------------------------------------------------
    # 2. Train Models
    # --------------------------------------------------------
    logger.info("🤖 Step 2 — Training Moneyline model...")
    ml_model, ml_meta = train_moneyline_model()

    logger.info("📈 Step 2 — Training Totals model...")
    totals_model, totals_meta = train_totals_model()

    logger.info("📉 Step 2 — Training Spread model...")
    spread_model, spread_meta = train_spread_model()

    # --------------------------------------------------------
    # 3. Register Models
    # --------------------------------------------------------
    logger.info("🗂️ Step 3 — Registering models in registry...")

    register_model("moneyline", ml_meta)
    register_model("totals", totals_meta)
    register_model("spread", spread_meta)

    logger.success("Models registered successfully.")

    # --------------------------------------------------------
    # 4. Run Predictions for Today
    # --------------------------------------------------------
    today = date.today()
    logger.info(f"🔮 Step 4 — Running predictions for {today}...")
    run_prediction_for_date(today)

    # --------------------------------------------------------
    # 5. Write Heartbeats
    # --------------------------------------------------------
    logger.info("❤️ Step 5 — Writing bootstrap + pipeline heartbeats...")
    _write_heartbeat(BOOTSTRAP_HEARTBEAT, "bootstrap")
    _write_heartbeat(PIPELINE_HEARTBEAT, "pipeline")

    logger.success("🎉 Full System Bootstrap Completed Successfully!")
    logger.info("Your system is now fully initialized and ready.")


def main():
    bootstrap_system()


if __name__ == "__main__":
    main()