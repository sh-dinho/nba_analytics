# File: config.py

import os
import toml
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Load TOML settings ---
SETTINGS_FILE = "settings.toml"
settings = {}
if os.path.exists(SETTINGS_FILE):
    settings = toml.load(SETTINGS_FILE)

# --- Directories ---
RESULTS_DIR = settings.get("general", {}).get("results_dir", "results")
DATA_DIR = settings.get("general", {}).get("data_dir", "data")
MODELS_DIR = settings.get("general", {}).get("models_dir", "models")

# --- File paths ---
CURRENT_FILE = os.path.join(RESULTS_DIR, "player_leaderboards_current.csv")
PREVIOUS_FILE = os.path.join(RESULTS_DIR, "player_leaderboards_previous.csv")
TRENDS_FILE = os.path.join(RESULTS_DIR, "player_trends.csv")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "weekly_summary.csv")
TEAM_SUMMARY_FILE = os.path.join(RESULTS_DIR, "team_summary.csv")
PICKS_FILE = os.path.join(RESULTS_DIR, "picks.csv")
PICKS_SUMMARY_FILE = os.path.join(RESULTS_DIR, "picks_summary.csv")

# --- Simulation defaults ---
INITIAL_BANKROLL = settings.get("general", {}).get("initial_bankroll", 1000)
DEFAULT_STRATEGY = settings.get("general", {}).get("default_strategy", "kelly")
DEFAULT_MAX_FRACTION = settings.get("general", {}).get("default_max_fraction", 0.05)
DEFAULT_THRESHOLD = settings.get("general", {}).get("probability_threshold", 0.65)
DEFAULT_MONTE_CARLO_SIMS = settings.get("general", {}).get("monte_carlo_sims", 1000)

# --- Notifications ---
TELEGRAM_ENABLED = settings.get("notifications", {}).get("telegram_enabled", True)
TELEGRAM_CHANNEL = os.getenv("TELEGRAM_CHANNEL", settings.get("notifications", {}).get("telegram_channel", "@nba_analytics_alerts"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# --- Odds API ---
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_URL = os.getenv("ODDS_API_URL", settings.get("odds_api", {}).get("api_url", "https://api.the-odds-api.com/v4/sports"))

# --- Streamlit settings ---
DASHBOARD_PAGES = settings.get("general", {}).get("pages", [
    "Daily Predictions",
    "Weekly Summary",
    "Weekly Player Trends",
    "CLI Results",
    "Monte Carlo Bankroll Simulation",
    "Player-Level Monte Carlo"
])

# --- Validation step ---
def validate_config():
    missing = []
    if not ODDS_API_KEY:
        missing.append("ODDS_API_KEY (odds provider API key)")
    if TELEGRAM_ENABLED and not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN (Telegram bot token)")
    if missing:
        print("⚠️ WARNING: Missing required configuration values:")
        for item in missing:
            print(f"   - {item}")
        print("Please set them in your .env file before running the app.")

# Run validation on import
validate_config()