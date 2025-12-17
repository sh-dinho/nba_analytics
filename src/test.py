import pytest
from pathlib import Path
import pandas as pd

from src import pipeline_runner

CACHE_DIR = Path("data/cache")


@pytest.mark.integration
def test_pipeline_creates_artifacts(monkeypatch):
    """
    Run the pipeline and check that key artifacts are created.
    """

    # Monkeypatch setup to use INFO logging but avoid clutter
    logger, cfg = pipeline_runner.setup()
    cfg.logging["level"] = "INFO"

    # Run pipeline
    pipeline_runner.run_pipeline()

    # Check that artifacts exist
    rankings_file = CACHE_DIR / "rankings.parquet"
    bets_file = CACHE_DIR / "bets.parquet"
    predictions_file = CACHE_DIR / "latest_predictions.parquet"

    assert rankings_file.exists(), "Rankings parquet not created"
    assert bets_file.exists(), "Bets parquet not created"
    assert predictions_file.exists(), "Latest predictions parquet not created"

    # Optionally check contents are non-empty
    rankings = pd.read_parquet(rankings_file)
    bets = pd.read_parquet(bets_file)
    preds = pd.read_parquet(predictions_file)

    assert not rankings.empty, "Rankings file is empty"
    assert not bets.empty, "Bets file is empty"
    assert not preds.empty, "Predictions file is empty"

    # Verify predictions contain identifiers and outcomes
    for col in ["homeTeam", "awayTeam", "predicted_outcome", "predicted_prob"]:
        assert col in preds.columns, f"Missing column {col} in predictions"
