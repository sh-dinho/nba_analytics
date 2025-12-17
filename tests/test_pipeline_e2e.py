# ============================================================
# File: tests/test_pipeline_e2e.py
# Purpose: End-to-end pipeline test (v2.0)
# Version: 2.0
# Author: Your Team
# Date: December 2025
# ============================================================

from pathlib import Path
import pandas as pd
from src.pipeline_runner import run_pipeline


def test_pipeline_end_to_end():
    # Execute pipeline
    run_pipeline()

    # Check enriched schedule saved
    enriched_parquet = Path("data/cache/master_schedule.parquet")
    assert enriched_parquet.exists()
    df = pd.read_parquet(enriched_parquet)
    assert not df.empty
    assert {"predicted_win", "predicted_outcome", "predicted_prob"}.issubset(df.columns)

    # Check rankings snapshot
    rankings_path = Path("data/cache/rankings.parquet")
    assert rankings_path.exists()

    # Check betting recommendations output (parquet or csv)
    csvs = list(Path("data/cache").glob("betting_recommendations_*.csv"))
    pars = list(Path("data/cache").glob("betting_recommendations_*.parquet"))
    assert csvs or pars

    # Check SHAP plots
    assert Path("logs/interpretability/shap_summary.png").exists()
    assert Path("logs/interpretability/shap_bar.png").exists()

    # Check archive created
    archives = list(Path("data/archive").glob("*"))
    assert archives, "Archive folder should contain a run directory"
