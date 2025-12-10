# NBA Prediction Pipeline

A Python pipeline for fetching NBA game data, generating features, training a logistic regression model, and producing daily win probability predictions. Outputs are saved locally in organized folders and can be connected directly to Power BI for dashboards.

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the pipeline
```bash
  python run_pipeline.py
```
3. Outputs will be saved automatically into the data/ folder structure.
```bash
nba_project/
├─ src/
│  ├─ __init__.py
│  ├─ utils/
│  │   ├─ __init__.py
│  │   ├─ io.py
│  │   ├─ logging.py
│  │   └─ nba_api_wrapper.py
│  ├─ feature_engineering/
│  │   ├─ __init__.py
│  │   └─ feature_engineering.py
│  ├─ model_training/
│  │   ├─ __init__.py
│  │   ├─ train_logreg.py
│  │   ├─ train_xgb.py
│  │   └─ training.py
│  ├─ prediction_engine/
│  │   ├─ __init__.py
│  │   └─ predictor.py
│  ├─ interpretability/
│  │   ├─ __init__.py
│  │   └─ shap_analysis.py
│  └─ main_today.py
├─ data/
├─ models/
├─ results/
└─ run_today.sh
```
4. Power Bi Integration
- Connect to Historical Prediction
  - Open Power BI Desktop
  - Go to Home -> Get Data -> Parquet
  - Select data/history/predictions_history.parquet
  - Load the table into Power BI.

5. Connect to Multiple Daily Files
  - Use the Folder connector:
    - for CSVs -> data/csv/
    - for Parquet -> data/parquet/
    - Load the table into Power BI

## Example Dashboards
  - Accuracy trend → Line chart with prediction_date vs. accuracy.
  - Team analytics → Bar chart with TEAM_ID vs. average pred_proba.
  - Game drill_downs → Table with stats + predictions.

6. 🛠 Features- Data Quality Checks → Validates critical columns, drops nulls, logs anomalies.
   - Error Handling → Retries API calls with exponential backoff, logs errors separately.
   - Configurable → Paths, seasons, and model path defined in config.yaml.
   - Environment Separation → Raw, cache, history, CSV, Parquet, logs all in distinct folders.
   - Deduplication → Unique IDs prevent duplicate rows.
   - Performance → Batch feature generation speeds up initial fetch.
   - Unit Tests → Core functions tested with pytest.

## 👥 Contributors- Developed in Python with ❤️ for NBA analytics.
  - Designed for easy integration with Power BI.

## 📈 Roadmap- v1.0 → Local Parquet/CSV storage, Power BI dashboards.
  - v2.0 → Optional migration to SQLite/PostgreSQL for larger datasets.
  - Future → Cloud integration (Azure Synapse, BigQuery, etc.).

## - Version 1.0 → stick with logistic regression + clean pipeline (done).
    - Version 2.0 → migrate storage to SQLite/Postgres.
    - Version 3.0 → add AI models (XGBoost or neural nets) and integrate explainability.

![Coverage](https://img.shields.io/codecov/c/github/your-org/your-repo?style=flat-square)
