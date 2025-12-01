# config.py
import os
import json

# --- Database Configuration ---
# Read path from ENV if available, otherwise use a default
DB_PATH = os.getenv("DB_PATH", "data/bets.db")

# --- API Configuration (Placeholders) ---
# NOTE: API keys should be set as environment variables in production, 
# but placeholders are defined here for structure.
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "YOUR_PLACEHOLDER_KEY")

# --- Model & Betting Constants ---
THRESHOLD = 0.6  # Minimum prediction probability for a strong pick
MAX_KELLY_FRACTION = 0.05 # Cap for Kelly staking

# --- Data Standardization ---
TEAM_ALIAS_PATH = "config/team_aliases.json"

def load_team_aliases() -> dict:
    """Loads the team mapping for standardization."""
    try:
        with open(TEAM_ALIAS_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Team alias file not found at {TEAM_ALIAS_PATH}. Using empty map.")
        return {}

TEAM_MAP = load_team_aliases()