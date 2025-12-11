# ============================================================
# File: src/main_today.py
# Purpose: Daily NBA predictions pipeline
# Version: 1.2 (robust imports, schema handling, SHAP dir creation)
# ============================================================

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.features.feature_engineering import generate_features_for_games
from src.prediction_engine.predictor import Predictor as NBAPredictor

# -----------------------------
# Load default config
# -----------------------------
CONFIG_FILE = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

DEFAULT_SCHEDULE_FILE = Path(config["paths"]["cache"]) / "schedule.parquet"
DEFAULT_MODEL_PATH = Path(config["model_path"])
DEFAULT_OUT_DIR = Path("results")

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Run daily NBA prediction pipeline")
parser.add_argument("--schedule_file", type=str, default=str(DEFAULT_SCHEDULE_FILE))
parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
parser.add_argument("--run_shap", action="store_true", help="Run SHAP interpretability analysis")

args = parser.parse_args()

SCHEDULE_FILE = Path(args.schedule_file)
MODEL_PATH = Path(args.model_path)
OUT_DIR = Path(args.out_dir)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("daily_runner")
logger.info("Starting NBA pipeline...")

# -----------------------------
# Load schedule/features
# -----------------------------
if not SCHEDULE_FILE.exists():
    logger.error(f"Schedule/features file not found: {SCHEDULE_FILE}")
    raise FileNotFoundError(f"{SCHEDULE_FILE} does not exist")

try:
    schedule_df = pd.read_parquet(SCHEDULE_FILE)
except Exception as e:
    logger.error(f"Failed to read schedule file: {e}")
    raise

if schedule_df.empty:
    logger.warning("Schedule/features file is empty")

# -----------------------------
# Generate features
# -----------------------------
logger.info("Generating features for today games...")
features_df = generate_features_for_games(schedule_df.to_dict(orient="records"))

# -----------------------------
# Load model and predict
# -----------------------------
logger.info(f"Loading model from {MODEL_PATH}")
predictor = NBAPredictor(model_path=str(MODEL_PATH))

logger.info("Predicting probabilities and labels...")
X = features_df.drop(columns=["win"], errors="ignore")
features_df["win_proba"] = predictor.predict_proba(X)
features_df["win_pred"] = predictor.predict_label(X)

# -----------------------------
# Save outputs
# -----------------------------
todays_csv = OUT_DIR / "todays_picks.csv"
features_df.to_csv(todays_csv, index=False)
logger.info(f"Predictions saved to {todays_csv}")

# Optional: run SHAP
if args.run_shap:
    from src.interpretability.shap_analysis import run_shap

    shap_dir = OUT_DIR / "interpretability"
    shap_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running SHAP analysis...")
    run_shap(str(MODEL_PATH), cache_file=str(SCHEDULE_FILE), out_dir=str(shap_dir))

logger.info("NBA pipeline finished.")
