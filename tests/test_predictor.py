import pytest
from src.prediction_engine.predictor import Predictor


@pytest.fixture
def mock_predictor():
    # Create an instance of the predictor class with a mock model path
    return Predictor(model_path="path/to/mock/model.pkl")


def test_predictor_initialization(mock_predictor):
    # Test that the predictor is initialized correctly
    assert mock_predictor is not None
    assert callable(
        mock_predictor.model
    )  # Ensure the model is callable (like a function)


def test_predictor_prediction(mock_predictor):
    # Test prediction functionality
    mock_data = {
        "TEAM_ID": 1,
        "OPPONENT_TEAM_ID": 2,
        "RollingPTS_5": 105,
        "RollingWinPct_10": 0.8,
    }

    prediction = mock_predictor.predict(mock_data)
    assert prediction in [0, 1]  # Assume binary outcome (win/loss)
