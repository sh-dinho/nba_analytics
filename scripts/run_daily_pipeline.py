# ============================================================
# File: scripts/run_daily_pipeline.py
# Purpose: Fully automated daily NBA betting simulation
# ============================================================

import pandas as pd
from datetime import date
from typing import List, Tuple

from config_loader import ConfigLoader
from fetch_features import fetch_nba_team_stats
from xgb_runner import xgb_runner
from core.paths import FEATURES_FILE, PICKS_FILE, PICKS_BANKROLL_FILE

# ------------------------------------------------------------
# 1. Load config & season info
# ------------------------------------------------------------
loader = ConfigLoader("config.toml")
season = loader.get_season("get-data", "2024-25")
if not loader.validate_season(season):
    raise ValueError("Season validation failed")

# ------------------------------------------------------------
# 2. Build API URL and fetch team stats
# ------------------------------------------------------------
url = loader.build_data_url(season)
print(f"Fetching NBA stats from: {url}")
df_stats = fetch_nba_team_stats(url)
df_stats.to_csv(FEATURES_FILE, index=False)

# ------------------------------------------------------------
# 3. Fetch today's games
# In a real pipeline, fetch today's games from NBA API or schedule CSV
# Here we simulate with an example subset
# ------------------------------------------------------------
today_games: List[Tuple[str, str]] = [
    ("Lakers", "Warriors"),
    ("Bulls", "Celtics")
]

# ------------------------------------------------------------
# 4. Fetch odds and OU lines (simulate or fetch from API)
# ------------------------------------------------------------
home_odds = [1.95, 2.05]
away_odds = [1.85, 1.90]
todays_games_uo = [220.5, 215.0]  # example over/under lines

# ------------------------------------------------------------
# 5. Build feature matrix aligned to today's games
# ------------------------------------------------------------
# Simplified: use stats directly; in production, engineer matchup features
feature_matrix = df_stats.iloc[:len(today_games)].values
frame_ml = df_stats.iloc[:len(today_games)].copy()

# ------------------------------------------------------------
# 6. Run XGBoost predictions and simulate bets
# ------------------------------------------------------------
results, history, metrics = xgb_runner(
    data=feature_matrix,
    todays_games_uo=todays_games_uo,
    frame_ml=frame_ml,
    games=today_games,
    home_team_odds=home_odds,
    away_team_odds=away_odds,
    use_kelly=True
)

# ------------------------------------------------------------
# 7. Store results
# ------------------------------------------------------------
pd.DataFrame(results).to_csv(PICKS_FILE, index=False)
pd.DataFrame({"bankroll": history}).to_csv(PICKS_BANKROLL_FILE, index=False)

print("✅ Daily pipeline complete!")
print(metrics)
