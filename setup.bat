@echo off
echo 🚀 Setting up NBA Analytics Project...

REM 1. Create folder structure
mkdir api
mkdir utils
mkdir models
mkdir logs

REM 2. Create virtual environment
python -m venv venv

REM 3. Activate virtual environment
call venv\Scripts\activate

REM 4. Upgrade pip
python -m pip install --upgrade pip

REM 5. Install dependencies
pip install pandas numpy xgboost scikit-learn requests schedule flask streamlit pyyaml beautifulsoup4 lxml

REM 6. Create requirements.txt
pip freeze > requirements.txt

REM 7. Initialize SQLite DB
python - <<END
import sqlite3
con = sqlite3.connect("nba_analytics.db")
cur = con.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS nba_games (
    Date TEXT, 
    Team TEXT, 
    Opponent TEXT, 
    HomeAway TEXT, 
    Result TEXT, 
    TeamScore INTEGER, 
    OpponentScore INTEGER
)""")
cur.execute("""CREATE TABLE IF NOT EXISTS daily_picks (
    Timestamp TEXT,
    Team TEXT,
    Opponent TEXT,
    Odds REAL,
    Probability REAL,
    EV REAL,
    SuggestedStake REAL
)""")
cur.execute("""CREATE TABLE IF NOT EXISTS bankroll_tracker (
    Timestamp TEXT,
    CurrentBankroll REAL,
    ROI REAL
)""")
cur.execute("""CREATE TABLE IF NOT EXISTS retrain_history (
    Timestamp TEXT,
    ModelType TEXT,
    Status TEXT
)""")
cur.execute("""CREATE TABLE IF NOT EXISTS model_metrics (
    Timestamp TEXT,
    Accuracy REAL,
    AUC REAL
)""")
con.commit()
con.close()
END

REM 8. Create placeholder config.yaml
echo server: > config.yaml
echo.  api_url: "http://127.0.0.1:5000" >> config.yaml
echo database: >> config.yaml
echo.  path: "nba_analytics.db" >> config.yaml
echo model: >> config.yaml
echo.  retrain_days: 7 >> config.yaml
echo notifications: >> config.yaml
echo.  telegram_token: "YOUR_TELEGRAM_BOT_TOKEN" >> config.yaml
echo.  telegram_chat_id: "YOUR_CHAT_ID" >> config.yaml

REM 9. Download sample XGBoost model
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/yourusername/nba_analytics/raw/main/xgb_model.pkl' -OutFile 'xgb_model.pkl'"

echo ✅ Setup complete! Activate environment using:
echo call venv\Scripts\activate
