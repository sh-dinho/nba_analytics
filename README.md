# 📘 NBA Prediction Pipeline — Clean & Production-Ready README (v1.3)
*A modular Python pipeline for fetching NBA game data, generating features, training ML models, and producing daily win-probability predictions. Fully compatible with Power BI.*

---

# 🚀 Quick Start

### **1. Install requirements**
```bash
pip install -r requirements.txt
```
### 2.Run the daily prediction runner
```bash
python run_pipeline.py --model models/nba_logreg.pkl
```
### 3.  Optional) Run the MLflow-enabled runner
```bash
python daily_runner_mflow.py --model models/nba_logreg.pkl
```
### 4. View outputs
All outputs are saved automatically into the standardized folder structure:
```bash
data/
  raw/           # raw NBA API dumps (optional)
  cache/         # cached training features
  history/       # historical predictions
  csv/           # daily CSV predictions
  parquet/       # daily Parquet predictions
  logs/          # runner logs + API failure logs
models/
results/
```
Your predictions are now ready for Power BI dashboards.

# 🏗 Project Structure

```
nba_analysis/
│
├── src/
│   ├── api/
│   │   └── nba_api_wrapper.py
│   ├── features/
│   ├── model_training/
│   ├── prediction_engine/
│   ├── tracker/
│   │   └── game_tracker.py
│   ├── utils/
│   │   ├── add_unique_id.py
│   │   ├── io.py
│   │   ├── logging.py
│   │   ├── logging_config.py
│   │   ├── mapping.py
│   │   ├── nba_api_wrapper.py
│   │   ├── validation.py
│   └── scripts/
│       ├── generate_historical_schedule.py
│       └── generate_today_schedule.py
├── data/
│   ├── cache/
│   └── results/
├── logs/
├── models/
├── tests/
├── docs/
├── .editorconfig
├── .gitignore
├── requirements.txt
├── setup_project.sh
└── Makefile

```
📊 Power BI Integration
1. Load Historical Prediction Data

Power BI → Get Data → Parquet

Select:
````
data/history/predictions_history.parquet
````
2. Load Multiple Daily Prediction Files

- Use the Folder connector:

  -For CSVs: data/csv/

- For Parquet: data/parquet/

Power BI automatically appends all files.

🛠 Key Pipeline Features
1. Data Quality Checks

validates required columns

ensures correct data types

detects anomalies

logs issues to data/logs/

2. Error Handling

automatic retry logic with backoff

safe API wrappers

separate error logs

3. Config-Driven

config.yaml controls:

seasons

model paths

thresholds

save locations

retry settings

MLflow parameters

4. File Structure Organization

Separate folders for:

raw API data

feature cache

prediction history

CSV & Parquet daily outputs

logs

5. Deduplication

Unified ID prevents duplicate rows:

GAME_ID

TEAM_ID

prediction_date

6. Performance

Vectorized feature engineering

Batch operations

Cached repeated lookups

7. Tested with pytest

Core components include tests:

feature generation

API wrapper

predictor logic

data cleaning

👥 Contributors

Developed in Python with ❤️ for NBA analytics, reproducible ML pipelines, and Power BI integration.

🗺 Roadmap
v1.0 — Complete

Logistic regression baseline

Clean pipeline

CSV/Parquet outputs

Power BI dashboards

v2.0 — Coming Soon

Migrate storage to SQLite/Postgres

Historical rollups

Scheduled ETL jobs

v3.0 — ML Enhancements

XGBoost / Random Forest / Neural Net models

SHAP explainability

MLflow model versioning

v4.0 — Cloud Integration

Azure Synapse

BigQuery

AWS Glue

cloud-based MLflow
