# Path: tests/test_imports.py
# Simple smoke tests to confirm that all core modules can be imported.

def test_import_data():
    from nba_analytics_core import data
    assert hasattr(data, "fetch_today_games")

def test_import_player_data():
    from nba_analytics_core import player_data
    assert hasattr(player_data, "fetch_player_season_stats")

def test_import_team_strength():
    from nba_analytics_core import team_strength
    assert hasattr(team_strength, "update_elo")

def test_import_awards():
    from nba_analytics_core import awards
    assert hasattr(awards, "train_mvp_classifier")