# tests/test_nba_api_client.py
import pandas as pd
import pytest
from types import SimpleNamespace
import src.api.nba_api_client as client


def test_fetch_season_games_schema(monkeypatch):
    class DummyLGF:
        def __init__(self, *args, **kwargs):
            pass

        def get_data_frames(self):
            return [
                pd.DataFrame(
                    {
                        "GAME_ID": ["001"],
                        "GAME_DATE": ["2025-12-11"],
                        "TEAM_NAME": ["LAL"],
                        "MATCHUP": ["LAL vs BOS"],
                    }
                )
            ]

    monkeypatch.setattr(client.leaguegamefinder, "LeagueGameFinder", DummyLGF)
    df = client.fetch_season_games("2025-26")
    assert list(df.columns) == client.EXPECTED_SEASON_COLS
    assert pd.api.types.is_datetime64_any_dtype(df["GAME_DATE"])


def test_fetch_today_games_parsing(monkeypatch):
    today = pd.Timestamp.now().date().strftime("%Y-%m-%d")

    class DummyLGF:
        def __init__(self, *args, **kwargs):
            pass

        def get_data_frames(self):
            return [
                pd.DataFrame(
                    {
                        "GAME_ID": ["001"],
                        "GAME_DATE": [today],
                        "TEAM_NAME": ["LAL"],
                        "MATCHUP": ["LAL @ BOS"],
                    }
                )
            ]

    monkeypatch.setattr(client.leaguegamefinder, "LeagueGameFinder", DummyLGF)
    df = client.fetch_today_games()
    assert "home_team" in df.columns and "away_team" in df.columns
    assert df.iloc[0]["home_team"] == "BOS"
    assert df.iloc[0]["away_team"] == "LAL"


def test_fetch_games_json_cache_and_parse(monkeypatch, tmp_path):
    # Mock responses for full schedule
    data = {
        "lscd": [
            {
                "mscd": {
                    "g": [
                        {
                            "gdte": "2025-12-11",
                            "gid": "002250001",
                            "h": {"tid": 1610612747},
                            "v": {"tid": 1610612738},
                        }
                    ]
                }
            }
        ]
    }
    # Redirect RAW_DIR
    client.RAW_DIR = str(tmp_path)

    def fake_get(url, timeout=10):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return data

        return Resp()

    monkeypatch.setattr(client.requests, "get", fake_get)
    df = client.fetch_games("2025-12-11", use_cache=False)
    assert set(df.columns) == set(client.EXPECTED_GAME_COLS)
    assert len(df) == 2  # home + away perspectives


def test_fetch_boxscores_missing_pd(monkeypatch, tmp_path):
    client.RAW_DIR = str(tmp_path)

    # Return JSON without 'pd'
    def fake_get(url, timeout=10):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"g": {}}

        return Resp()

    monkeypatch.setattr(client.requests, "get", fake_get)
    df = client.fetch_boxscores(["002250001"], use_cache=False)
    assert df.empty
    assert list(df.columns) == client.EXPECTED_BOX_COLS
