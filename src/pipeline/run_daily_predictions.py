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
#       - Ensures season schedule + canonical schedule are fresh
#       - Ensures LONG_SNAPSHOT has rows for today (fallback)
#       - Uses tricodes everywhere (team, opponent)
#       - Gracefully handles empty or failing modules
# ============================================================

from datetime import date
from typing import Optional

import pandas as pd
from loguru import logger

from src.config.paths import LONG_SNAPSHOT, COMBINED_PRED_DIR
from src.ingestion.schedule_refresh import refresh_schedule_if_needed
from src.ingestion.fallback_schedule import ensure_long_snapshot_has_date
from src.model.predict import run_prediction_for_date as run_moneyline_predictions
from src.model.predict_totals import run_totals_prediction_for_date
from src.model.predict_spread import run_spread_prediction_for_date


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def _load_long_rows_for_date(pred_date: date) -> pd.DataFrame:
    """
    Load team-game rows for pred_date from LONG_SNAPSHOT.
    """
    if not LONG_SNAPSHOT.exists():
        logger.warning(f"[Orchestrator] LONG_SNAPSHOT missing at {LONG_SNAPSHOT}")
        return pd.DataFrame()

    df = pd.read_parquet(LONG_SNAPSHOT)
    if "date" not in df.columns:
        logger.error(
            f"[Orchestrator] LONG_SNAPSHOT at {LONG_SNAPSHOT} missing 'date' column."
        )
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    todays = df[df["date"].dt.date == pred_date].copy()

    logger.info(
        f"[Orchestrator] Loaded {len(todays)} long-format rows for {pred_date} "
        f"from LONG_SNAPSHOT."
    )
    return todays


def _ensure_long_rows_for_date(pred_date: date) -> pd.DataFrame:
    """
    Ensure we have long-format team-game rows for pred_date in LONG_SNAPSHOT.
    If missing, synthesize them via fallback_schedule.
    """
    todays = _load_long_rows_for_date(pred_date)
    if not todays.empty:
        return todays

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
    else:
        logger.info(
            f"[Orchestrator] Using fallback long-format rows for {pred_date}: "
            f"rows={len(todays)}."
        )

    return todays


def _safe_run_moneyline(pred_date: date) -> pd.DataFrame:
    """
    Run moneyline predictions, but never let model-layer errors crash the orchestrator.
    """
    try:
        logger.info(f"[Orchestrator] Starting moneyline predictions for {pred_date}")
        ml_df = run_moneyline_predictions(pred_date)

        if ml_df.empty:
            logger.warning(
                "[Orchestrator] Moneyline predictions returned empty DataFrame."
            )
            return pd.DataFrame()

        # Prefer home rows if is_home exists; otherwise keep as-is
        if "is_home" in ml_df.columns:
            ml_home = ml_df[ml_df["is_home"] == 1].copy()
            if ml_home.empty:
                logger.warning(
                    "[Orchestrator] Moneyline predictions have 'is_home' but no "
                    "home rows. Using full frame."
                )
                ml_home = ml_df.copy()
        else:
            logger.warning(
                "[Orchestrator] Moneyline predictions missing 'is_home' column. "
                "Using full frame without home/away split."
            )
            ml_home = ml_df.copy()

        # Normalize naming for merge if team/opponent exist
        if {"team", "opponent"}.issubset(ml_home.columns):
            ml_home = ml_home.rename(
                columns={"team": "home_team", "opponent": "away_team"}
            )

        logger.info(
            f"[Orchestrator] Moneyline predictions ready: rows={len(ml_home)}, "
            f"cols={len(ml_home.columns)}"
        )
        return ml_home

    except Exception as e:
        logger.error(f"[Orchestrator] Moneyline predictions failed: {e}")
        return pd.DataFrame()


def _safe_run_totals(pred_date: date) -> pd.DataFrame:
    try:
        logger.info(f"[Orchestrator] Starting totals predictions for {pred_date}")
        totals_df = run_totals_prediction_for_date(pred_date)
        if totals_df.empty:
            logger.warning(
                "[Orchestrator] Totals predictions returned empty DataFrame."
            )
            return pd.DataFrame()

        logger.info(
            f"[Orchestrator] Totals predictions ready: rows={len(totals_df)}, "
            f"cols={len(totals_df.columns)}"
        )
        return totals_df
    except Exception as e:
        logger.error(f"[Orchestrator] Totals predictions failed: {e}")
        return pd.DataFrame()


def _safe_run_spread(pred_date: date) -> pd.DataFrame:
    try:
        logger.info(f"[Orchestrator] Starting spread predictions for {pred_date}")
        spread_df = run_spread_prediction_for_date(pred_date)
        if spread_df.empty:
            logger.warning(
                "[Orchestrator] Spread predictions returned empty DataFrame."
            )
            return pd.DataFrame()

        logger.info(
            f"[Orchestrator] Spread predictions ready: rows={len(spread_df)}, "
            f"cols={len(spread_df.columns)}"
        )
        return spread_df
    except Exception as e:
        logger.error(f"[Orchestrator] Spread predictions failed: {e}")
        return pd.DataFrame()


def _merge_predictions(
    ml_home: pd.DataFrame,
    totals_df: pd.DataFrame,
    spread_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """
    Merge the three prediction DataFrames on game_id where possible.
    """
    if ml_home.empty and totals_df.empty and spread_df.empty:
        logger.warning(
            "[Orchestrator] All prediction modules returned empty. Nothing to merge."
        )
        return None

    combined: Optional[pd.DataFrame] = None

    # Choose a base frame
    if not ml_home.empty:
        combined = ml_home.copy()
        logger.info("[Orchestrator] Using moneyline predictions as merge base.")
    elif not totals_df.empty:
        combined = totals_df.copy()
        logger.info("[Orchestrator] Using totals predictions as merge base.")
    elif not spread_df.empty:
        combined = spread_df.copy()
        logger.info("[Orchestrator] Using spread predictions as merge base.")

    # Merge totals
    if (
        combined is not None
        and not totals_df.empty
        and "game_id" in combined.columns
        and "game_id" in totals_df.columns
    ):
        combined = combined.merge(
            totals_df[
                [
                    "game_id",
                    "predicted_total_points",
                ]
            ],
            on="game_id",
            how="left",
        )
        logger.info("[Orchestrator] Merged totals predictions into combined frame.")

    # Merge spread
    if (
        combined is not None
        and not spread_df.empty
        and "game_id" in combined.columns
        and "game_id" in spread_df.columns
    ):
        spread_cols = []
        if "predicted_margin" in spread_df.columns:
            spread_cols.append("predicted_margin")
        if "spread_line" in spread_df.columns:
            spread_cols.append("spread_line")

        combined = combined.merge(
            spread_df[["game_id"] + spread_cols],
            on="game_id",
            how="left",
        )
        logger.info("[Orchestrator] Merged spread predictions into combined frame.")

    return combined


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------


def run_daily_predictions(
    pred_date: date | None = None,
    auto_ingest: bool = False,  # reserved for future use
) -> pd.DataFrame:
    pred_date = pred_date or date.today()
    logger.info(f"🏀 Running daily predictions orchestrator for {pred_date}")

    # Step 0 — Ensure schedule is fresh (season + canonical snapshot)
    refresh_schedule_if_needed(pred_date)

    # Step 1 — Ensure we have team-game rows for today (API or fallback)
    todays_long = _ensure_long_rows_for_date(pred_date)
    if todays_long.empty:
        # Nothing to do
        logger.warning(
            f"[Orchestrator] No long-format rows for {pred_date} after fallback. "
            "Skipping predictions."
        )
        return pd.DataFrame()

    # --------------------------------------------------------
    # Step 2 — Run individual prediction modules (safely)
    # --------------------------------------------------------
    ml_home = _safe_run_moneyline(pred_date)
    totals_df = _safe_run_totals(pred_date)
    spread_df = _safe_run_spread(pred_date)

    # --------------------------------------------------------
    # Step 3 — Merge predictions
    # --------------------------------------------------------
    combined = _merge_predictions(ml_home, totals_df, spread_df)

    # --------------------------------------------------------
    # Step 4 — Save combined predictions
    # --------------------------------------------------------
    if combined is not None and not combined.empty:
        COMBINED_PRED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = COMBINED_PRED_DIR / f"combined_{pred_date}.parquet"
        combined.to_parquet(out_path, index=False)
        logger.success(f"[Orchestrator] Combined predictions saved → {out_path}")
    else:
        logger.warning("[Orchestrator] Combined predictions empty — nothing saved.")
        combined = pd.DataFrame()

    logger.success(f"🏁 Daily predictions pipeline complete for {pred_date}")
    return combined


if __name__ == "__main__":
    run_daily_predictions()
