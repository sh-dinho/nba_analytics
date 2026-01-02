from __future__ import annotations
from pathlib import Path
import json

# ============================================================
# 🏀 NBA Analytics — Path Configuration
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

DAILY_SCHEDULE_SNAPSHOT = CANONICAL_DIR / "schedule_daily.parquet"
LONG_SNAPSHOT = CANONICAL_DIR / "long_snapshot.parquet"
FEATURES_SNAPSHOT = CANONICAL_DIR / "features_snapshot.parquet"
SEASON_SCHEDULE_PATH = CANONICAL_DIR / "schedule_season.parquet"

# ------------------------------------------------------------
# Raw snapshots
# ------------------------------------------------------------
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_FEATURE_SNAPSHOT_PATH = SNAPSHOTS_DIR / "features_snapshot.parquet"

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
# Models + registry
# ------------------------------------------------------------
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REGISTRY_DIR = DATA_DIR / "model_registry"
MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REGISTRY_PATH = MODEL_REGISTRY_DIR / "index.json"

# Ensure registry is valid JSON
if not MODEL_REGISTRY_PATH.exists():
    MODEL_REGISTRY_PATH.write_text('{"models": []}', encoding="utf-8")
else:
    try:
        data = json.loads(MODEL_REGISTRY_PATH.read_text())
        if "models" not in data:
            raise ValueError
    except Exception:
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
