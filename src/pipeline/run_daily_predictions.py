from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Daily Predictions Orchestrator
# File: src/pipeline/run_daily_predictions.py
#
# Description:
#     Runs all prediction pipelines (moneyline, totals, spread)
#     for a given date and merges them into a combined output.
#
#     Behaviors:
#       - Ensures season_schedule.parquet is fresh
#       - Ensures LONG_SNAPSHOT has rows for today (fallback)
#       - Uses tricodes everywhere (team, opponent)
#       - Gracefully handles empty modules
# ============================================================

from datetime import date
import pandas as pd
from loguru import logger

from src.config.paths import (
    LONG_SNAPSHOT,
    COMBINED_PRED_DIR,
)
from src.model.predict import run_prediction_for_date as run_moneyline_predictions
from src.model.predict_totals import run_totals_prediction_for_date
from src.model.predict_spread import run_spread_prediction_for_date
from src.ingestion.schedule_refresh import refresh_schedule_if_needed
from src.ingestion.fallback_schedule import ensure_long_snapshot_has_date


def _load_rows_for_date(pred_date: date) -> pd.DataFrame:
    """
    Load team-game rows for pred_date from LONG_SNAPSHOT.
    """
    if not LONG_SNAPSHOT.exists():
        logger.warning(f"[Orchestrator] LONG_SNAPSHOT missing at {LONG_SNAPSHOT}")
        return pd.DataFrame()

    df = pd.read_parquet(LONG_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"])
    todays = df[df["date"].dt.date == pred_date].copy()
    return todays


def _ensure_ingestion(pred_date: date, auto_ingest: bool = False):
    """
    Placeholder for ingestion logic.
    In v4, ingestion is typically run separately.
    """
    if not auto_ingest:
        logger.info(
            f"[Orchestrator] Skipping ingestion step for {pred_date} (auto_ingest=False)."
        )
        return


def run_daily_predictions(
    pred_date: date | None = None, auto_ingest: bool = False
) -> pd.DataFrame:
    pred_date = pred_date or date.today()
    logger.info(f"🏀 Running daily predictions orchestrator for {pred_date}")

    # Step 0 — Ensure schedule is fresh
    refresh_schedule_if_needed(max_age_days=1)

    # Step 1 — Optional ingestion
    _ensure_ingestion(pred_date, auto_ingest=auto_ingest)

    # Step 2 — Ensure we have team-game rows for today (API or fallback)
    todays = _load_rows_for_date(pred_date)
    if todays.empty:
        logger.warning(
            f"[Orchestrator] LONG_SNAPSHOT has no rows for {pred_date}. "
            f"Attempting fallback schedule synthesis."
        )
        todays = ensure_long_snapshot_has_date(pred_date)

    if todays.empty:
        logger.warning(
            f"[Orchestrator] No NBA games available in LONG_SNAPSHOT or fallback "
            f"for {pred_date}. Skipping all predictions."
        )
        return pd.DataFrame()

    # --------------------------------------------------------
    # Step 3 — Moneyline predictions
    # --------------------------------------------------------
    ml_df = run_moneyline_predictions(pred_date)
    if ml_df.empty:
        logger.warning("[Orchestrator] Moneyline predictions returned empty DataFrame.")
        ml_home = pd.DataFrame()
    else:
        if "is_home" not in ml_df.columns:
            logger.error(
                "[Orchestrator] Moneyline predictions missing 'is_home' column."
            )
            ml_home = pd.DataFrame()
        else:
            ml_home = ml_df[ml_df["is_home"] == 1].rename(
                columns={"team": "home_team", "opponent": "away_team"}
            )

    # --------------------------------------------------------
    # Step 4 — Totals predictions
    # --------------------------------------------------------
    totals_df = run_totals_prediction_for_date(pred_date)
    if totals_df.empty:
        logger.warning("[Orchestrator] Totals predictions returned empty DataFrame.")

    # --------------------------------------------------------
    # Step 5 — Spread predictions
    # --------------------------------------------------------
    spread_df = run_spread_prediction_for_date(pred_date)
    if spread_df.empty:
        logger.warning("[Orchestrator] Spread predictions returned empty DataFrame.")

    # --------------------------------------------------------
    # Step 6 — Merge predictions
    # --------------------------------------------------------
    if ml_home.empty and totals_df.empty and spread_df.empty:
        logger.warning(
            "[Orchestrator] All prediction modules returned empty. Nothing to merge."
        )
        return pd.DataFrame()

    combined = None

    # Start with moneyline (home rows)
    if not ml_home.empty:
        combined = ml_home.copy()
    else:
        if not totals_df.empty:
            combined = totals_df.copy()
        elif not spread_df.empty:
            combined = spread_df.copy()

    # Merge totals
    if combined is not None and not totals_df.empty and "game_id" in totals_df.columns:
        combined = combined.merge(
            totals_df[["game_id", "predicted_total_points"]],
            on="game_id",
            how="left",
        )

    # Merge spread
    if combined is not None and not spread_df.empty and "game_id" in spread_df.columns:
        combined = combined.merge(
            spread_df[["game_id", "predicted_margin"]],
            on="game_id",
            how="left",
        )

    # --------------------------------------------------------
    # Step 7 — Save combined predictions
    # --------------------------------------------------------
    if combined is not None and not combined.empty:
        COMBINED_PRED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = COMBINED_PRED_DIR / f"combined_{pred_date}.parquet"
        combined.to_parquet(out_path, index=False)
        logger.success(f"[Orchestrator] Combined predictions saved → {out_path}")
    else:
        logger.warning("[Orchestrator] Combined predictions empty — nothing saved.")

    logger.success(f"🏁 Daily predictions pipeline complete for {pred_date}")
    return combined


if __name__ == "__main__":
    run_daily_predictions()
