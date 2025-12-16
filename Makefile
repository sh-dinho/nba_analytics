# ============================================================
# NBA Analysis Pipeline Makefile (Optimized)
# ============================================================

SEASON ?= 2025
YEARS ?= 3
DATE ?= $(shell date +%Y-%m-%d)

MASTER = data/reference/historical_master.parquet

.PHONY: baseline refresh ingest features train predict today all clean rebuild

# --- Baseline: download schedule for current season
baseline:
	python -m src.scripts.download_baseline_schedule --season $(SEASON)

# --- Refresh: incremental update of master historical dataset
refresh: baseline
	python -m src.scripts.refresh_schedule_incremental --season $(SEASON)

# --- Ingest: build features from master file
ingest: refresh
	python -m src.data.ingest --end-season $(shell expr $(SEASON) - 1) --years $(YEARS) --refresh

# --- Feature generation
features: ingest
	python -m src.scripts.generate_features --season $(SEASON)

# --- Train model
train: features
	python -m src.model.train --season $(SEASON) --type logreg

# --- Predict daily games
predict: train
	python -m src.daily_runner.daily_runner_mflow --date $(DATE) --model models/nba_logreg.pkl --enable-shap

# --- Generate today's schedule
today:
	python -m src.scripts.generate_today_schedule --date $(DATE)

# --- Full pipeline
all: baseline refresh ingest features train predict

# --- Cleanup
clean:
	rm -f data/cache/*.csv
	rm -f data/cache/*.parquet
	rm -f data/reference/*.csv
	rm -f data/reference/*.parquet
	rm -f data/logs/*.log

rebuild: clean
