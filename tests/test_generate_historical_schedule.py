# ============================================================
# File: tests/test_generate_historical_schedule.py
# Purpose: Validate historical schedule generation script
# Project: nba_analysis
# ============================================================

import pandas as pd
import pytest
import os

import src.scripts.generate_historical_schedule as script


class DummyGameFinder:
    """Mock LeagueGameFinder returning a simple DataFrame."""

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
                    "PTS": [100],
                    "WL": ["W"],
                }
            )
        ]


def test_successful_fetch(monkeypatch, tmp_path):
    # Patch LeagueGameFinder to return dummy data
    monkeypatch.setattr(
        script,
        "leaguegamefinder",
        type("LGF", (), {"LeagueGameFinder": DummyGameFinder}),
    )
    # Patch OUTPUT_FILE to temporary location
    script.OUTPUT_FILE = tmp_path / "historical_schedule.parquet"

    script.main()

    # Verify file created
    assert script.OUTPUT_FILE.exists()
    df = pd.read_parquet(script.OUTPUT_FILE)
    assert "GAME_ID" in df.columns
    assert len(df) == 1


def test_error_fetch(monkeypatch, tmp_path):
    # Patch LeagueGameFinder to raise error
    def bad_finder(*args, **kwargs):
        raise Exception("API error")

    monkeypatch.setattr(
        script, "leaguegamefinder", type("LGF", (), {"LeagueGameFinder": bad_finder})
    )
    script.OUTPUT_FILE = tmp_path / "historical_schedule.parquet"

    # Run main, should not create file
    script.main()
    assert not script.OUTPUT_FILE.exists()


def test_no_games(monkeypatch, tmp_path):
    # Patch LeagueGameFinder to return empty DataFrame
    class EmptyGameFinder:
        def __init__(self, *args, **kwargs):
            pass

        def get_data_frames(self):
            return [
                pd.DataFrame(
                    columns=[
                        "GAME_DATE",
                        "TEAM_NAME",
                        "MATCHUP",
                        "GAME_ID",
                        "TEAM_ID",
                        "PTS",
                        "WL",
                    ]
                )
            ]

    monkeypatch.setattr(
        script,
        "leaguegamefinder",
        type("LGF", (), {"LeagueGameFinder": EmptyGameFinder}),
    )
    script.OUTPUT_FILE = tmp_path / "historical_schedule.parquet"

    script.main()
    # File should not exist because no games were fetched
    assert not script.OUTPUT_FILE.exists()
