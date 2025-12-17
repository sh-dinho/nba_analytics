# 🏀 NBA Analytics Pipeline v2.3

## Overview
This project is a **modern SaaS analytics platform for NBA game outcome prediction and betting insights**.  
It orchestrates ingestion, schema alignment, feature engineering, model training, explainability, rankings, betting recommendations, and artifact archiving — all with CI/CD integration and automated notifications.

Key highlights:
- 📥 **Data ingestion** from NBA boxscore schemas
- 🧾 **Schema normalization & validation**
- ⚙️ **Feature engineering** (Elo ratings, rolling stats, opponent-adjusted metrics, rest days)
- 🤖 **Model training & prediction** (RandomForest + extensible ensemble support)
- 🔍 **Explainability** with SHAP plots
- 📊 **Rankings & betting recommendations**
- 📦 **Artifact management** with versioned archives
- 📲 **Notifications** via Telegram/Slack
- 🛡 **CI/CD** with linting, header enforcement, and automated tests

---

## 📂 Project Structure
```
src/
config/ # YAML configs + loader
 features/ # Feature engineering (Elo, rolling, opponent-adjusted) 
 models/ # Model training & prediction 
 ranking/ # Betting recommendations
  schedule/ # Historical pipeline + schema contracts
   schemas/ # Normalization logic 
   utils/ # Logging, IO helpers pipeline_runner.py # Main orchestration script 
   scripts/ add_headers.py # Header enforcement utility 
   tests/ test_feature_engineering.py # Unit tests for features test_pipeline_e2e.py # End-to-end pipeline tests
    data/ 
      history/ # Raw historical NBA data
      cache/ # Latest enriched schedule, rankings, recs
       archive/ # Versioned run artifacts 
       logs/ interpretability/ # SHAP plots 
       .github/workflows/ pipeline.yml # CI/CD workflow 
  requirements.txt # Python dependencies
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.11+
- Virtual environment recommended

### Install dependencies
```bash
pip install -r requirements.txt
```
Prepare historical data
Place a parquet file in data/history/historical_schedule.parquet with NBA boxscore schema:
```
SEASON_ID, TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME,
GAME_ID, GAME_DATE, MATCHUP, WL, PTS, ...
```
The pipeline automatically aligns this schema to canonical:
```
gameId, seasonYear, startDate, homeTeam, awayTeam, homeScore, awayScore
```
Run pipeline
```
python -m src.pipeline_runner
```
This will:

Ingest & align historical data

Enrich schedule with features

Train & predict outcomes

Generate SHAP explainability plots

Produce rankings & betting recommendations

Archive artifacts with metadata

📊 Outputs
Enriched schedule → data/cache/master_schedule.parquet

Rankings → data/cache/rankings.parquet

Betting recommendations → data/cache/betting_recommendations_YYYY-MM-DD.parquet

Explainability plots → logs/interpretability/shap_summary.png, shap_bar.png

Archived artifacts → data/archive/<timestamp>/

🛡 CI/CD
Linting: flake8 + black

Header enforcement: scripts/add_headers.py

Pipeline run: executes end-to-end

Notifications: Telegram alerts with top picks + SHAP plots

🧪 Testing
Unit tests: Elo ratings, rolling features, opponent-adjusted metrics, schema alignment

End-to-end tests: Full pipeline run from ingestion → archiving

Numerical validation: Elo updates, rolling averages correctness

📅 Roadmap (2026)
Q1: Feature store, drift detection, schema validation

Q2: Ensemble models, Bayesian updating, analyst notes

Q3: Automated retraining, MLflow model registry, richer notifications

Q4: Player-level features, betting market integration, interactive dashboards

👤 Author
Developed by  (sh) — Architect and lead developer of a modern SaaS analytics platform for sports betting.

📜 License
MIT License (or specify your chosen license).
