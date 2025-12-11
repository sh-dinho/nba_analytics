# ============================================================
# File: tests/test_predictor_cli.py
# Purpose: Unit tests for predictor_cli.py
# ============================================================

import sys
import pandas as pd
import pytest
from click.testing import CliRunner

import src.prediction_engine.predictor_cli as predictor_cli


@pytest.fixture
def runner():
    return CliRunner()


def test_no_games(monkeypatch, runner):
    monkeypatch.setattr(
        predictor_cli, "fetch_season_games", lambda season, limit: pd.DataFrame()
    )
    result = runner.invoke(
        predictor_cli.cli, ["--model", "fake.pkl", "--season", "2025"]
    )
    assert "❌ No games found." in result.output


def test_proba_default(monkeypatch, runner):
    games_df = pd.DataFrame([{"game_id": 1}, {"game_id": 2}])
    features_df = pd.DataFrame({"feat_a": [0.1, 0.2], "win": [1, 0]})

    monkeypatch.setattr(
        predictor_cli, "fetch_season_games", lambda season, limit: games_df
    )
    monkeypatch.setattr(
        predictor_cli, "generate_features_for_games", lambda games: features_df
    )

    class DummyPredictor:
        def __init__(self, model_path):
            pass

        def predict_proba(self, X):
            return pd.Series([0.7, 0.3])

    monkeypatch.setattr(predictor_cli, "NBAPredictor", DummyPredictor)

    result = runner.invoke(
        predictor_cli.cli, ["--model", "fake.pkl", "--season", "2025"]
    )
    assert "win_proba" in result.output
    assert "0.7" in result.output or "0.3" in result.output


def test_label(monkeypatch, runner):
    games_df = pd.DataFrame([{"game_id": 1}, {"game_id": 2}])
    features_df = pd.DataFrame({"feat_a": [0.1, 0.2], "win": [1, 0]})

    monkeypatch.setattr(
        predictor_cli, "fetch_season_games", lambda season, limit: games_df
    )
    monkeypatch.setattr(
        predictor_cli, "generate_features_for_games", lambda games: features_df
    )

    class DummyPredictor:
        def __init__(self, model_path):
            pass

        def predict_label(self, X):
            return pd.Series([1, 0])

    monkeypatch.setattr(predictor_cli, "NBAPredictor", DummyPredictor)

    result = runner.invoke(
        predictor_cli.cli, ["--model", "fake.pkl", "--season", "2025", "--label"]
    )
    assert "win_pred" in result.output
    assert "1" in result.output or "0" in result.output


def test_error(monkeypatch, runner):
    def fail_fetch(season, limit):
        raise RuntimeError("API failure")

    monkeypatch.setattr(predictor_cli, "fetch_season_games", fail_fetch)

    result = runner.invoke(
        predictor_cli.cli, ["--model", "fake.pkl", "--season", "2025"]
    )
    assert "❌ Prediction run failed" in result.output
