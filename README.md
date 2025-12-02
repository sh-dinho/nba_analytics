# 🏀 NBA Analytics Pipeline

A fully automated NBA analytics pipeline that fetches team stats, merges data, runs predictions, simulates bankroll with EV + Kelly criterion, and sends daily reports to Telegram.

---

## 🚀 Features
- **Data ingestion**: Fetches NBA team stats from [stats.nba.com](https://stats.nba.com) using season ranges in `config.toml`.
- **Database merge**: Consolidates season CSVs into `teamdata_all.sqlite`.
- **Prediction models**: Logistic regression, neural networks, and XGBoost.
- **Bankroll simulation**: Calculates Expected Value (EV), Kelly bet sizes, and bankroll trajectory.
- **Telegram reporting**: Sends formatted daily reports with emojis (EV 🟢🔴, bankroll 📈📉).
- **Automation**: One command (`scripts/run_pipeline.py`) or scheduled GitHub Action runs the entire pipeline.
- **Archiving & logging**: Season CSVs archived automatically, logs written to `logs/pipeline.log`.

---

## 📂 Project Structure
nba_analytics/ 
app/ prediction_pipeline.py   # Main prediction pipeline scripts/ fetch_season_data.py  # Fetch season stats merge_team_data       # Merge CSVs into SQLite run_pipeline.py        
# Orchestration script telegram_report.py       
# Send daily report to Telegram sbr_odds_provider.py     
# Odds provider stub config.toml               
# Season configuration results/                   
# Predictions + bankroll CSVs data/seasons/              
# Raw season CSVs archive/seasons/           
# Archived CSVs logs/pipeline.log        
# Rotating pipeline logs


---

## ⚙️ Setup

1. **Clone repo**
   ```bash
   git clone https://github.com/yourname/nba_analytics.git
   cd nba_analytics
   pip install -r requirements.txt
- Set environment variables
- TELEGRAM_TOKEN → your bot token
- TELEGRAM_CHAT_ID → your chat/group ID
Run pipeline manually
python scripts/run_pipeline.py


---

## ✅ Outcome
- **Workflow**: CI/CD automation runs daily, sends Telegram report.
- **README**: Clear documentation for setup, usage, automation, and outputs.
- Everything is now **self‑contained, automated, and documented**.

---
