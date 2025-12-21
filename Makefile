# Makefile for NBA Analytics v3
PYTHON := python

.PHONY: full fetch_data run_pipeline dashboard monitor clean

# Default: Runs the whole data flow
full: fetch_data run_pipeline
	@echo "✅ Full NBA pipeline finished!"

# Step 0: Data Acquisition
fetch_data:
	@echo "⏳ Fetching latest NBA games..."
	$(PYTHON) -m src.scripts.fetch_real_nba_data

# Step 1-5: The Core Pipeline
run_pipeline:
	@echo "⏳ Running full ML pipeline..."
	$(PYTHON) -m src.pipeline.pipeline_runner

# Launch the Streamlit UI
dashboard:
	@echo "🚀 Launching Streamlit Dashboard..."
	streamlit run streamlit_app.py

# Launch Prometheus and Grafana (requires Docker)
monitor:
	@echo "📊 Spinning up Monitoring Stack..."
	docker-compose up -d
	@echo "Grafana: http://localhost:3000 | Prometheus: http://localhost:9090"

# Cleanup intermediate files
clean:
	@echo "🧹 Cleaning data directories..."
	rm -rf data/ingestion/*.parquet
	rm -rf data/predictions/*.parquet
	rm -rf data/rankings/*.parquet