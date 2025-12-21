# ============================================================
# Project: NBA Analytics & Betting Engine
# Author: Sadiq
# Description: Execute recommended bets safely, log them, and
#              prepare for integration with simulators or books.
# ============================================================

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from src.config.betting import BANKROLL
from src.config.paths import LOGS_DIR


REQUIRED_RECOMMENDATION_COLUMNS = [
    "game_id",
    "market_team",
    "market_side",
    "ml",
    "model_prob",
    "implied_prob",
    "edge",
    "decimal_odds",
    "ev_per_unit",
    "kelly_fraction",
    "recommended_stake",
    "confidence",
    "prediction_date",
    "model_version",
    "feature_version",
]


def _validate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the recommendations DataFrame is valid and safe to execute.
    """
    if df is None or df.empty:
        logger.info("auto_bet(): received empty recommendations.")
        return pd.DataFrame()

    missing = set(REQUIRED_RECOMMENDATION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"auto_bet(): missing required columns: {missing}")

    # Drop zero or negative stakes
    before = len(df)
    df = df[df["recommended_stake"] > 0].copy()
    dropped = before - len(df)
    if dropped:
        logger.info(f"auto_bet(): dropped {dropped} rows with non-positive stakes.")

    return df


def _load_bet_log() -> pd.DataFrame:
    """
    Load existing bet log or create a new one.
    """
    path = LOGS_DIR / "bets.parquet"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "game_id",
                "market_team",
                "market_side",
                "ml",
                "stake",
                "model_prob",
                "implied_prob",
                "edge",
                "ev_per_unit",
                "kelly_fraction",
                "confidence",
                "prediction_date",
                "model_version",
                "feature_version",
            ]
        )
    return pd.read_parquet(path)


def _save_bet_log(df: pd.DataFrame):
    """
    Save the updated bet log.
    """
    path = LOGS_DIR / "bets.parquet"
    df.to_parquet(path, index=False)
    logger.success(f"auto_bet(): updated bet log → {path}")


def execute_bets(
    recommendations: pd.DataFrame,
    bankroll: float = BANKROLL,
    dry_run: bool = True,
) -> pd.DataFrame:
    """
    Execute recommended bets.

    Parameters:
        recommendations: output of recommend_bets()
        bankroll: bankroll used for sizing (for logging only)
        dry_run: if True, no real betting integration is attempted

    Returns:
        DataFrame of executed bets (logged)
    """
    logger.info("=== Auto-Bet Execution Start ===")

    df = _validate_recommendations(recommendations)
    if df.empty:
        logger.info("auto_bet(): no bets to execute.")
        return df

    # Load existing log
    log_df = _load_bet_log()

    executed_rows = []

    for _, row in df.iterrows():
        game_id = row["game_id"]
        stake = float(row["recommended_stake"])
        ml = row["ml"]

        # Dry-run mode: no external integration
        if dry_run:
            logger.info(
                f"[DRY RUN] Would place bet: game_id={game_id}, "
                f"{row['market_team']} ({row['market_side']}), "
                f"ML={ml}, stake={stake:.2f}"
            )
        else:
            # Placeholder for future sportsbook integration
            logger.info(
                f"[LIVE] Placing bet: game_id={game_id}, "
                f"{row['market_team']} ({row['market_side']}), "
                f"ML={ml}, stake={stake:.2f}"
            )

        executed_rows.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "game_id": row["game_id"],
                "market_team": row["market_team"],
                "market_side": row["market_side"],
                "ml": row["ml"],
                "stake": stake,
                "model_prob": row["model_prob"],
                "implied_prob": row["implied_prob"],
                "edge": row["edge"],
                "ev_per_unit": row["ev_per_unit"],
                "kelly_fraction": row["kelly_fraction"],
                "confidence": row["confidence"],
                "prediction_date": row["prediction_date"],
                "model_version": row["model_version"],
                "feature_version": row["feature_version"],
            }
        )

    # Append to log
    executed_df = pd.DataFrame(executed_rows)
    log_df = pd.concat([log_df, executed_df], ignore_index=True)

    _save_bet_log(log_df)

    logger.success(
        f"auto_bet(): executed {len(executed_df)} bets "
        f"(dry_run={dry_run}, bankroll={bankroll})."
    )
    logger.info("=== Auto-Bet Execution End ===")

    return executed_df
