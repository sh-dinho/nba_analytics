from __future__ import annotations
from pathlib import Path

# ============================================================
# 🏀 NBA Analytics v4
# Module: Path Configuration
# File: src/config/paths.py
#
# Description:
#     Centralized paths for all data artifacts:
#       - canonical snapshots
#       - ingestion cache
#       - model artifacts + registry
#       - predictions (moneyline, totals, spread, combined)
#       - odds
#       - reports
#       - orchestrator logs
#
#     v4 alignment:
#       - LONG_SNAPSHOT is the single source of truth for
#         feature building (training + prediction)
#       - SCHEDULE_SNAPSHOT stores game-level schedule only
#       - All directories self-create
# ============================================================


# ------------------------------------------------------------
# Root directory
# ------------------------------------------------------------

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

# Long-format snapshot (team-game rows)
LONG_SNAPSHOT = CANONICAL_DIR / "long.parquet"

# Game-level schedule snapshot (home/away rows)
SCHEDULE_SNAPSHOT = CANONICAL_DIR / "schedule.parquet"

# Full season schedule (game-level, from external source e.g., Basketball Reference)
SEASON_SCHEDULE_PATH = CANONICAL_DIR / "season_schedule.parquet"

# ------------------------------------------------------------
# Ingestion cache
# ------------------------------------------------------------

INGESTION_CACHE_DIR = DATA_DIR / "cache"
INGESTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Scoreboard cache (v3/v4 ingestion)
SCOREBOARD_CACHE_DIR = INGESTION_CACHE_DIR / "scoreboard"
SCOREBOARD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REGISTRY_INDEX = DATA_DIR / "model_registry" / "index.json"
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

# Initialize registry if missing
if not MODEL_REGISTRY_PATH.exists():
    MODEL_REGISTRY_PATH.write_text('{"models": []}', encoding="utf-8")


# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

PREDICTIONS_DIR = DATA_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Moneyline predictions (team rows)
MONEYLINE_PRED_DIR = DATA_DIR / "predictions_moneyline"
MONEYLINE_PRED_DIR.mkdir(parents=True, exist_ok=True)

# Totals predictions (home rows)
TOTALS_PRED_DIR = DATA_DIR / "predictions_totals"
TOTALS_PRED_DIR.mkdir(parents=True, exist_ok=True)

# Spread predictions (home rows)
SPREAD_PRED_DIR = DATA_DIR / "predictions_spread"
SPREAD_PRED_DIR.mkdir(parents=True, exist_ok=True)

# Combined predictions (merged ML + totals + spread)
COMBINED_PRED_DIR = DATA_DIR / "predictions_combined"
COMBINED_PRED_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Odds (future sportsbook ingestion)
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