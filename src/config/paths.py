from __future__ import annotations
from pathlib import Path

# ============================================================
# 🏀 NBA Analytics v4
# Module: Path Configuration
# File: src/config/paths.py
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------
# Data root
# ------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Canonical snapshots
# ------------------------------------------------------------
CANONICAL_DIR = DATA_DIR / "canonical"
CANONICAL_DIR.mkdir(parents=True, exist_ok=True)

LONG_SNAPSHOT = CANONICAL_DIR / "long.parquet"
SCHEDULE_SNAPSHOT = CANONICAL_DIR / "schedule.parquet"
SEASON_SCHEDULE_PATH = CANONICAL_DIR / "season_schedule.parquet"

# ------------------------------------------------------------
# Ingestion cache
# ------------------------------------------------------------
INGESTION_CACHE_DIR = DATA_DIR / "cache"
INGESTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SCOREBOARD_CACHE_DIR = INGESTION_CACHE_DIR / "scoreboard"
SCOREBOARD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Features
# ------------------------------------------------------------
FEATURES_DIR = DATA_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Model artifacts + registry
# ------------------------------------------------------------
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REGISTRY_DIR = DATA_DIR / "model_registry"
MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REGISTRY_PATH = MODEL_REGISTRY_DIR / "index.json"

if not MODEL_REGISTRY_PATH.exists():
    MODEL_REGISTRY_PATH.write_text('{"models": []}', encoding="utf-8")

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------
PREDICTIONS_DIR = DATA_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

MONEYLINE_PRED_DIR = DATA_DIR / "predictions_moneyline"
MONEYLINE_PRED_DIR.mkdir(parents=True, exist_ok=True)

TOTALS_PRED_DIR = DATA_DIR / "predictions_totals"
TOTALS_PRED_DIR.mkdir(parents=True, exist_ok=True)

SPREAD_PRED_DIR = DATA_DIR / "predictions_spread"
SPREAD_PRED_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_PRED_DIR = DATA_DIR / "predictions_combined"
COMBINED_PRED_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Odds
# ------------------------------------------------------------
ODDS_DIR = DATA_DIR / "odds"
ODDS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Reports
# ------------------------------------------------------------
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Orchestrator logs
# ------------------------------------------------------------
ORCHESTRATOR_LOG_DIR = DATA_DIR / "orchestrator_logs"
ORCHESTRATOR_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Bet Tracker (NEW)
# ------------------------------------------------------------
BET_LOG_DIR = DATA_DIR / "bets"
BET_LOG_DIR.mkdir(parents=True, exist_ok=True)

BET_LOG_PATH = BET_LOG_DIR / "bet_log.csv"

# ------------------------------------------------------------
# Streamlit App (NEW)
# ------------------------------------------------------------
STREAMLIT_APP_DIR = ROOT_DIR / "src" / "app"
STREAMLIT_APP_DIR.mkdir(parents=True, exist_ok=True)

# Bet Tracker
BET_LOG_DIR = DATA_DIR / "bets"
BET_LOG_DIR.mkdir(parents=True, exist_ok=True)
BET_LOG_PATH = BET_LOG_DIR / "bet_log.csv"

# Odds
ODDS_DIR = DATA_DIR / "odds"
ODDS_DIR.mkdir(parents=True, exist_ok=True)
