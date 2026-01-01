from __future__ import annotations
from pathlib import Path

# ============================================================
# 🏀 NBA Analytics
# Module: Path Configuration
# File: src/config/paths.py
# Author: Sadiq
#
# Description:
#     Centralized directory and file path configuration for:
#       • canonical snapshots
#       • ingestion cache
#       • features
#       • models + registry
#       • predictions (moneyline, spread, totals, combined)
#       • odds
#       • results
#       • backtesting
#       • reports
#       • monitoring logs + dashboards
#       • bet tracker
#       • recommendations
#       • Streamlit app
# ============================================================

# ------------------------------------------------------------
# Root
# ------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------
# Data root
# ------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Canonical snapshots (version-agnostic)
# ------------------------------------------------------------
CANONICAL_DIR = DATA_DIR / "canonical"
CANONICAL_DIR.mkdir(parents=True, exist_ok=True)

SCHEDULE_SNAPSHOT = CANONICAL_DIR / "schedule_snapshot.parquet"
LONG_SNAPSHOT = CANONICAL_DIR / "long_snapshot.parquet"
FEATURES_SNAPSHOT = CANONICAL_DIR / "features_snapshot.parquet"

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
# Results
# ------------------------------------------------------------
RESULTS_SNAPSHOT_DIR = DATA_DIR / "results"
RESULTS_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_SNAPSHOT_PATH = RESULTS_SNAPSHOT_DIR / "results.parquet"

# ------------------------------------------------------------
# Backtesting
# ------------------------------------------------------------
BACKTEST_DIR = DATA_DIR / "backtest"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Reports
# ------------------------------------------------------------
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Monitoring logs + dashboards
# ------------------------------------------------------------
LOGS_DIR = DATA_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DASHBOARD_DIR = DATA_DIR / "dashboard"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

DASHBOARD_RECOMMENDATIONS_PATH = DASHBOARD_DIR / "recommendations.json"
DASHBOARD_BANKROLL_PATH = DASHBOARD_DIR / "bankroll.json"

# ------------------------------------------------------------
# Orchestrator logs
# ------------------------------------------------------------
ORCHESTRATOR_LOG_DIR = DATA_DIR / "orchestrator_logs"
ORCHESTRATOR_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Bet Tracker
# ------------------------------------------------------------
BET_LOG_DIR = DATA_DIR / "bets"
BET_LOG_DIR.mkdir(parents=True, exist_ok=True)

BET_LOG_PATH = BET_LOG_DIR / "bet_log.csv"

# ------------------------------------------------------------
# Recommendations
# ------------------------------------------------------------
RECOMMENDATIONS_DIR = DATA_DIR / "recommendations"
RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------
STREAMLIT_APP_DIR = ROOT_DIR / "src" / "app"
STREAMLIT_APP_DIR.mkdir(parents=True, exist_ok=True)

RAW_FEATURE_SNAPSHOT_PATH = DATA_DIR / "snapshots" / "features_snapshot.parquet"