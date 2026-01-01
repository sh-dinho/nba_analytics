from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Bet Settlement
# File: src/betting/bet_settlement.py
# Author: Sadiq
#
# Description:
#     Settle executed bets using final game results.
#     Adds:
#       • result (win/loss)
#       • payout
#       • profit
# ============================================================

import pandas as pd
from loguru import logger

from src.config.paths import BET_LOG_PATH, BET_LOG_DIR, RESULTS_SNAPSHOT_DIR


# ------------------------------------------------------------
# Loaders
# ------------------------------------------------------------
def _load_bet_log() -> pd.DataFrame:
    if not BET_LOG_PATH.exists():
        logger.warning("bet_settlement(): no bet log found.")
        return pd.DataFrame()
    return pd.read_csv(BET_LOG_PATH)


def _load_results() -> pd.DataFrame:
    """
    Load canonical results snapshot.
    Must contain:
        game_id, team, won, decimal_odds (or closing odds)
    """
    try:
        df = pd.read_parquet(RESULTS_SNAPSHOT_DIR / "results.parquet")
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    except Exception as e:
        logger.error(f"Failed to load results snapshot: {e}")
        return pd.DataFrame()


# ------------------------------------------------------------
# Settlement
# ------------------------------------------------------------
def settle_bets() -> pd.DataFrame:
    """
    Settle all unsettled bets in the bet log.
    Adds:
        result: "win" | "loss"
        payout: stake * (odds - 1) if win else 0
        profit: payout - stake
    """
    log_df = _load_bet_log()
    if log_df.empty:
        return log_df

    results = _load_results()
    if results.empty:
        return log_df

    # Ensure settlement columns exist
    for col in ["result", "payout", "profit"]:
        if col not in log_df.columns:
            log_df[col] = None

    unsettled = log_df[log_df["result"].isna()].copy()
    if unsettled.empty:
        logger.info("bet_settlement(): no unsettled bets.")
        return log_df

    # Merge on game_id + market_team
    merged = unsettled.merge(
        results[["game_id", "team", "won"]],
        left_on=["game_id", "market_team"],
        right_on=["game_id", "team"],
        how="left",
        suffixes=("", "_res"),
    )

    # If no result found, skip settlement
    merged = merged[merged["won"].notna()].copy()
    if merged.empty:
        logger.warning("bet_settlement(): no matching results for unsettled bets.")
        return log_df

    # Compute settlement fields
    merged["result"] = merged["won"].apply(lambda x: "win" if x == 1 else "loss")
    merged["payout"] = merged.apply(
        lambda r: r["stake"] * (r["decimal_odds"] - 1) if r["result"] == "win" else 0.0,
        axis=1,
    )
    merged["profit"] = merged["payout"] - merged["stake"]

    # Update original log using correct index mapping
    for original_idx, row in merged.iterrows():
        log_df.loc[original_idx, ["result", "payout", "profit"]] = (
            row["result"],
            row["payout"],
            row["profit"],
        )

    # Save updated log
    BET_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(BET_LOG_PATH, index=False)

    logger.success(f"bet_settlement(): settled {len(merged)} bets.")
    return log_df