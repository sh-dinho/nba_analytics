from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Auto-Bet Execution
# File: src/betting/auto_bet.py
# Author: Sadiq
#
# Description:
#     Execute recommended bets safely, log them, and prepare
#     for integration with sportsbooks or simulators.
# ============================================================

from datetime import datetime
import pandas as pd
from loguru import logger

from src.config.betting import BANKROLL
from src.config.paths import BET_LOG_PATH, BET_LOG_DIR


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


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------
def _validate(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        logger.info("auto_bet(): empty recommendations.")
        return pd.DataFrame()

    missing = set(REQUIRED_RECOMMENDATION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"auto_bet(): missing required columns: {missing}")

    # Remove zero or negative stakes
    df = df[df["recommended_stake"] > 0].copy()

    # Remove NaN stakes
    df = df[df["recommended_stake"].notna()]

    return df


# ------------------------------------------------------------
# Log loading + saving
# ------------------------------------------------------------
def _load_log() -> pd.DataFrame:
    if not BET_LOG_PATH.exists():
        BET_LOG_DIR.mkdir(parents=True, exist_ok=True)
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
    return pd.read_csv(BET_LOG_PATH)


def _save_log(df: pd.DataFrame):
    BET_LOG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(BET_LOG_PATH, index=False)
    logger.success(f"auto_bet(): updated bet log → {BET_LOG_PATH}")


# ------------------------------------------------------------
# Execution
# ------------------------------------------------------------
def execute_bets(
    recommendations: pd.DataFrame,
    bankroll: float = BANKROLL,
    dry_run: bool = True,
) -> pd.DataFrame:
    """
    Execute recommended bets and append them to the bet log.

    Parameters
    ----------
    recommendations : pd.DataFrame
        Must contain REQUIRED_RECOMMENDATION_COLUMNS.
    bankroll : float
        Current bankroll (not modified here, but logged for context).
    dry_run : bool
        If True, no real bets are placed.

    Returns
    -------
    pd.DataFrame
        Executed bets (one row per bet).
    """

    logger.info("=== Auto-Bet Execution Start ===")

    df = _validate(recommendations)
    if df.empty:
        logger.info("auto_bet(): no bets to execute.")
        return df

    log_df = _load_log()
    executed = []

    for _, row in df.iterrows():
        stake = float(row["recommended_stake"])

        # Safety guard
        if stake <= 0 or pd.isna(stake):
            logger.warning(f"Skipping invalid stake: {stake}")
            continue

        if dry_run:
            logger.info(
                f"[DRY RUN] Would place bet: {row['market_team']} ({row['market_side']}), "
                f"ML={row['ml']}, stake={stake:.2f}"
            )
        else:
            logger.info(
                f"[LIVE] Placing bet: {row['market_team']} ({row['market_side']}), "
                f"ML={row['ml']}, stake={stake:.2f}"
            )

        executed.append(
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

    executed_df = pd.DataFrame(executed)
    log_df = pd.concat([log_df, executed_df], ignore_index=True)

    _save_log(log_df)

    logger.success(
        f"auto_bet(): executed {len(executed_df)} bets "
        f"(dry_run={dry_run}, bankroll={bankroll})."
    )
    logger.info("=== Auto-Bet Execution End ===")

    return executed_df