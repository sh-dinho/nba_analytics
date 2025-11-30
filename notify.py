import requests
import pandas as pd
from datetime import date
import yaml
import logging
import os

logging.basicConfig(level=logging.INFO)

# === Load Config Safely ===
def load_config(path="config.yaml"):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise RuntimeError("⚠️ config.yaml not found. Please create one with nba_api_key, telegram_token, chat_id.")

config = load_config()
NBA_API_KEY = config.get("nba_api_key")
TELEGRAM_TOKEN = config.get("telegram_token")
CHAT_ID = config.get("chat_id")

# === Fetch Today’s Games ===
def fetch_today_games():
    today = date.today().strftime("%Y-%m-%d")
    url = f"https://api.sportsdata.io/v4/nba/scores/json/GamesByDate/{today}?key={NBA_API_KEY}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"❌ Failed to fetch games: {e}")

    data = resp.json()
    if not data:
        logging.info("ℹ️ No NBA games scheduled today.")
        return pd.DataFrame()

    games = []
    for g in data:
        games.append({
            "team1": g.get("HomeTeam"),
            "team2": g.get("AwayTeam"),
            "season": g.get("Season"),
            "status": g.get("Status"),
            "start_time": g.get("DateTime")
        })
    return pd.DataFrame(games)

# === Predict Outcomes (placeholder AI logic) ===
def predict_matchup(team1, team2, season):
    # Replace with your trained classifier
    return 0.55, 0.45

def predict_over_under(team1, team2, season):
    # Replace with your regression model
    return 220

# === Generate Recommendations ===
def generate_recommendations(bankroll=1000):
    games = fetch_today_games()
    if games.empty:
        return pd.DataFrame()

    recs = []
    for _, g in games.iterrows():
        team1, team2, season = g["team1"], g["team2"], g["season"]
        prob1, prob2 = predict_matchup(team1, team2, season)
        total_points = predict_over_under(team1, team2, season)

        recs.append({
            "matchup": f"{team1} vs {team2}",
            "team1_prob": prob1,
            "team2_prob": prob2,
            "expected_total": total_points
        })
    return pd.DataFrame(recs)

# === Send to Telegram ===
def send_telegram_recommendations(df):
    if df.empty:
        logging.info("ℹ️ No recommendations to send.")
        return

    text = "📊 Today’s NBA Recommendations:\n\n"
    for _, row in df.iterrows():
        team1, team2 = row['matchup'].split(' vs ')
        text += f"{row['matchup']}\n"
        text += f" - {row['team1_prob']:.2%} chance for {team1}\n"
        text += f" - {row['team2_prob']:.2%} chance for {team2}\n"
        text += f" - Expected total points: {row['expected_total']}\n\n"

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        resp.raise_for_status()
        logging.info("✅ Recommendations sent to Telegram")
    except requests.RequestException as e:
        raise RuntimeError(f"❌ Telegram send failed: {e}")

# === MAIN ===
if __name__ == "__main__":
    df = generate_recommendations()
    send_telegram_recommendations(df)