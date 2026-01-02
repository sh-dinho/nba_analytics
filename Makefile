# ============================================================
# NBA Analytics Engine — Makefile
# ============================================================

# Cross-platform Python resolver:
# - Uses python3 if available
# - Falls back to python on Windows
PYTHON := $(shell command -v python3 >/dev/null 2>&1 && echo python3 || echo python)

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

lint:
	ruff check .

format:
	black .
	isort .

typecheck:
	mypy src/

test:
	pytest -q

ingest:
	$(PYTHON) -m src.scripts.run_ingestion

predict:
	$(PYTHON) -m src.scripts.predict_today

monitor:
	$(PYTHON) -m src.scripts.monitor_daily

dashboard:
	$(PYTHON) -m src.scripts.generate_data_quality_dashboard

backtest:
	$(PYTHON) -m src.scripts.run_backtest_report --range 30