import pytest
from src.prediction_engine.predictor import Predictor  # Update this path as needed

@pytest.fixture
def mock_predictor():
    # Adjust this to match the actual Predictor constructor
    # If Predictor expects a model instance or a different method to load the model, change this.
    predictor = Predictor()  # If model loading is done later or via another method
    predictor.load_model(model_path="path/to/mock/model.pkl")  # Use the actual method to load the model
    return predictor

def test_predictor_initialization(mock_predictor):
    # Test that the predictor is initialized correctly
    assert mock_predictor is not None

def test_predictor_prediction(mock_predictor):
    # Test prediction functionality
    mock_data = {
        'TEAM_ID': 1,
        'OPPONENT_TEAM_ID': 2,
        'RollingPTS_5': 105,
        'RollingWinPct_10': 0.8
    }

    prediction = mock_predictor.predict(mock_data)
    assert prediction in [0, 1]  # Assume binary outcome (win/loss)
