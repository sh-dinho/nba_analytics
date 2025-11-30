import requests
import logging

TELEGRAM_TOKEN = ""
TELEGRAM_CHAT_ID = ""
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

def send_telegram_message(text):
    """Send message to Telegram chat."""
    try:
        response = requests.post(f"{BASE_URL}/sendMessage", json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text
        })
        response.raise_for_status()
        logging.info("Message sent to Telegram.")
    except Exception as e:
        logging.error(f"Error sending message to Telegram: {e}")


def send_daily_picks():
    """Send daily picks for NBA games to Telegram."""
    try:
        con = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM nba_games WHERE date = CURRENT_DATE", con)
        con.close()
    except Exception as e:
        logging.error(f"DB read error: {e}")
        return

    if df.empty:
        send_telegram_message("❌ No games today.")
        return

    # Load the trained model
    model = joblib.load(MODEL_PATH)

    # Make predictions
    picks = []
    for _, row in df.iterrows():
        home_score = row['home_score']
        away_score = row['away_score']
        home_team = row['home_team']
        away_team = row['visitor_team']
        
        # Feature preparation
        home_team_win_pct = home_score / (home_score + away_score)
        away_team_win_pct = away_score / (home_score + away_score)

        # Prediction
        prediction = model.predict([[home_score, away_score, home_team_win_pct, away_team_win_pct]])[0]
        picks.append({
            "home_team": home_team,
            "away_team": away_team,
            "prediction": "Home win" if prediction == 1 else "Away win"
        })

    # Send picks to Telegram
    msg = "💰 --- TODAY'S PICKS --- 💰\n"
    for pick in picks:
        msg += f"\n{pick['home_team']} vs {pick['away_team']} | Prediction: {pick['prediction']}"
    send_telegram_message(msg)
