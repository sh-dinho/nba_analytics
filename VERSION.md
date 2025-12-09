# NBA Prediction Pipeline — Version Roadmap
---
✅ Current State (v1.0)
- Pipeline: Fetches NBA games, generates features, trains logistic regression, predicts daily outcomes.
- Storage: Organized folders (raw, cache, history, csv, parquet, logs, models).
- Config: Centralized in config.yaml.
- Quality: Data validation, deduplication, error handling with retries.
- Performance: Batch feature generation.
- Docs: README.md + VERSION.md roadmap.
- Dependencies: requirements.txt ensures reproducibility.
---
🚀 Planned Enhancements
v1.1
- Add SHAP explainability (already drafted).
- Power BI dashboard with global feature importance + game drilldowns.
- Expanded logging (structured JSON).
- CI/CD setup for automated runs.
---
v2.0
- Database integration (SQLite/Postgres).
- Modular restructure (data_ingestion.py, feature_engineering.py, train_model.py, predict.py).
- Power BI connected directly to DB.
---
v3.0
- AI upgrades:
- XGBoost (already drafted).
- SHAP explainability integrated.
- Player props (20+ points, rebounds, assists).
- Spread (+/-) and totals (over/under).
- Tracking modules:
- Top 6 teams per conference.
- Top 6 players per conference.
- Teams to bet on / avoid.
- Winning streaks and hot players.
---
📊 Dashboard Expansion
- Win/Loss tab → baseline predictions.
- Spread tab → cover probabilities.
- Totals tab → over/under probabilities.
- Player Props tab → per‑player milestones.
- Rankings tab → top 6 teams/players.
- Betting Insights tab → bet/avoid recommendations.
- Trends tab → streaks, hot players, feature importance shifts.
---
