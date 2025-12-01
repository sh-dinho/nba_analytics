# README.md
# NBA Analytics Core

Core tools for NBA data ingestion, feature engineering, predictions, and bankroll simulation.

## Quickstart
- Create and initialize DB:
  - python scripts/init_db.py
- Update DB with seasons and export feature stats:
  - python scripts/update_db.py
- Run daily pipeline:
  - nba-pipeline
- Launch dashboard:
  - streamlit run app/app.py

## Config
Edit config.yaml and environment variables (DB_PATH, ODDS_API_KEY, PREDICTION_THRESHOLD).

## Tests
pytest