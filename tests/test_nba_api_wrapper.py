# ============================================================
# File: tests/test_nba_api_wrapper.py
# Purpose: Validate NBA API wrapper utilities
# Project: nba_analysis
# ============================================================

import pandas as pd
import pytest
from src.utils import nba_api_wrapper


class DummyGameFinder:
    def __init__(self, *args, **kwargs):
        pass

    def get_data_frames(self):
        return [
            pd.DataFrame(
                {
                    "GAME_DATE": ["2025-01-01"],
                    "TEAM_NAME": ["TestTeam"],
                    "MATCHUP": ["TestTeam vs Opponent"],
                    "GAME_ID": ["001"],
                    "TEAM_ID": [123],
                    "OPPONENT_TEAM_ID": [456],
                    "PTS": [100],
                    "WL": ["W"],
                }
            )
        ]


class DummyScoreboard:
    def __init__(self, *args, **kwargs):
        pass

    def get_data_frames(self):
        return [
            pd.DataFrame(
                {"GAME_ID": ["001"], "HOME_TEAM_ID": [123], "VISITOR_TEAM_ID": [456]}
            )
        ]


def test_fetch_season_games_success(monkeypatch):
    monkeypatch.setattr(
        nba_api_wrapper.leaguegamefinder, "LeagueGameFinder", DummyGameFinder
    )
    df = nba_api_wrapper.fetch_season_games("2025-26")
    assert not df.empty
    assert "POINTS" in df.columns
    assert "TARGET" in df.columns
    assert df.loc[0, "TARGET"] == 1  # W mapped to 1


def test_fetch_season_games_invalid_type():
    with pytest.raises(TypeError):
        nba_api_wrapper.fetch_season_games(2025)


def test_fetch_season_games_error(monkeypatch):
    def bad_finder(*args, **kwargs):
        raise Exception("API error")

    monkeypatch.setattr(
        nba_api_wrapper.leaguegamefinder, "LeagueGameFinder", bad_finder
    )
    df = nba_api_wrapper.fetch_season_games("2025-26")
    assert list(df.columns) == nba_api_wrapper.EXPECTED_COLS


def test_fetch_today_games_success(monkeypatch):
    monkeypatch.setattr(nba_api_wrapper.scoreboard, "Scoreboard", DummyScoreboard)
    df = nba_api_wrapper.fetch_today_games()
    assert not df.empty
    assert "AWAY_TEAM_ID" in df.columns


def test_fetch_today_games_error(monkeypatch):
    def bad_scoreboard(*args, **kwargs):
        raise Exception("API error")

    monkeypatch.setattr(nba_api_wrapper.scoreboard, "Scoreboard", bad_scoreboard)
    df = nba_api_wrapper.fetch_today_games()
    assert list(df.columns) == nba_api_wrapper.EXPECTED_TODAY_COLS


def test_fetch_games_today(monkeypatch):
    monkeypatch.setattr(nba_api_wrapper.scoreboard, "Scoreboard", DummyScoreboard)
    df = nba_api_wrapper.fetch_games()
    assert "AWAY_TEAM_ID" in df.columns


def test_fetch_games_season(monkeypatch):
    monkeypatch.setattr(
        nba_api_wrapper.leaguegamefinder, "LeagueGameFinder", DummyGameFinder
    )
    df = nba_api_wrapper.fetch_games("2025-26")
    assert "POINTS" in df.columns
