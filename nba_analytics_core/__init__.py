# Path: nba_analytics_core/__init__.py
# This file marks the folder as a Python package.
# You don’t need to put anything inside, but you can expose modules if you want.

# Example (optional):
from .data import fetch_historical_games, fetch_today_games, build_team_stats, build_matchup_features
from .player_data import fetch_player_season_stats, build_player_leaderboards
from .team_strength import update_elo, championship_probabilities
from .awards import train_mvp_classifier, predict_mvp