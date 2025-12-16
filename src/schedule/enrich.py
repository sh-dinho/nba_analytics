# ============================================================
# File: src/schedule/enrich.py
# Purpose: Enrich and validate TEAM-level schedule
# Project: nba_analysis
# ============================================================

import pandas as pd
from pathlib import Path
from src.utils.common import configure_logging, save_dataframe, load_dataframe
from src.schemas.normalize import normalize
from src.schedule.contract import validate_team_schedule

logger = configure_logging(name="schedule.enrich")


def enrich_schedule(baseline_path: str, output_prefix: str) -> pd.DataFrame:
    """Validate baseline schedule, normalize, and save enriched outputs."""
    if not Path(baseline_path).exists():
        logger.error("Baseline schedule not found at %s", baseline_path)
        return pd.DataFrame()

    df = load_dataframe(baseline_path)
    if df.empty:
        logger.error("Baseline schedule is empty.")
        return pd.DataFrame()

    df = normalize(df, "enriched_schedule")
    validate_team_schedule(df)

    save_dataframe(df, f"{output_prefix}.csv")
    save_dataframe(df, f"{output_prefix}.parquet")
    logger.info(
        "Enriched schedule saved to %s.[csv|parquet] (rows=%d)", output_prefix, len(df)
    )
    return df
