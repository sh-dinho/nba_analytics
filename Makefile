# ============================================================
# File: Makefile
# Purpose: Unified task runner for NBA AI project
# ============================================================

.PHONY: default install check ci clean docs serve_docs train features mlflow help test run

INVOKE = invoke

# --- Default target ---
default: ci

# --- Bootstrap project ---
install:
	./setup_project.sh

# --- Local validation pipeline ---
check:
	$(INVOKE) check

# --- Continuous Integration pipeline ---
ci:
	$(INVOKE) ci

# --- Clean caches and temp files ---
clean:
	$(INVOKE) clean

# --- Build documentation ---
docs:
	$(INVOKE) docs

# --- Serve documentation locally ---
serve_docs:
	$(INVOKE) serve_docs

# --- Model training ---
train:
	$(INVOKE) train

# --- Feature generation ---
features:
	$(INVOKE) features

# --- Launch MLflow UI ---
mlflow:
	$(INVOKE) mlflow

# --- Run pipeline ---
run:
	python src/run_pipeline.py

# --- Run tests ---
test:
	$(INVOKE) test

# --- Show help ---
help:
	@echo "Available targets:"
	@echo "  install       Bootstrap project"
	@echo "  check         Run local validation pipeline"
	@echo "  ci            Run CI pipeline"
	@echo "  clean         Clean caches and temp files"
	@echo "  docs          Build documentation"
	@echo "  serve_docs    Serve documentation locally"
	@echo "  train         Train model"
	@echo "  features      Generate features"
	@echo "  mlflow        Launch MLflow UI"
	@echo "  run           Run pipeline script"
	@echo "  test          Run unit tests"
