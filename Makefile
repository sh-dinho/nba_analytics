# ============================================================
# Project: nba_analysis
# Purpose: Automate end-to-end pipeline (enrich → features → train → predict)
# ============================================================

PYTHON := python

ENRICHED_FILE := data/cache/historical_schedule_with_results.parquet
FEATURES_FILE := data/cache/features_full.parquet
MODEL_FILE := models/nba_xgb.pkl
PREDICTIONS_FILE := data/results/daily_predictions.csv

# Default target
run_all: enrich generate_features train predict

# Step 0: Enrich schedule with WL outcomes (skip future games)
enrich:
	$(PYTHON) -m src.scripts.enrich_schedule

# Step 1: Generate features from enriched schedule
generate_features:
	$(PYTHON) -m src.scripts.generate_features

# Step 2: Train model
train:
	$(PYTHON) -m src.model_training.trainer_cli \
		--model xgb \
		--season 2025 \
		--features $(FEATURES_FILE) \
		--out $(MODEL_FILE)

# Step 3: Run daily predictions
predict:
	$(PYTHON) -m src.prediction_engine.daily_runner_cli \
		--model $(MODEL_FILE) \
		--season 2025 \
		--limit 10 \
		--out $(PREDICTIONS_FILE) \
		--fmt csv

# Clean generated files
clean:
	rm -f $(ENRICHED_FILE) $(FEATURES_FILE) $(MODEL_FILE) $(PREDICTIONS_FILE)
