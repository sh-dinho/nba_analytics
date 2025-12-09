# ============================================================
# File: Makefile
# Purpose: Unified task runner for NBA AI project
# ============================================================

.PHONY: install precommit check ci clean docs serve_docs train features mlflow

# --- Bootstrap project ---
install:
	./setup_project.sh

# --- Pre-commit hooks ---
precommit:
	pre-commit run --all-files

# --- Local validation pipeline ---
check:
	invoke check

# --- Continuous Integration pipeline ---
ci:
	invoke ci

# --- Clean caches and temp files ---
clean:
	invoke clean

# --- Build documentation ---
docs:
	invoke docs

# --- Serve documentation locally ---
serve_docs:
	invoke serve_docs

# --- Model training ---
train:
	invoke train

# --- Feature generation ---
features:
	invoke features

# --- Launch MLflow UI ---
mlflow:
	invoke mlflow
