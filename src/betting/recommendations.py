from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Unified Recommendation Pipeline
# File: src/betting/recommendations.py
# Author: Sadiq
#
# Description:
#     High-level wrapper that ties together:
#       • value bet construction
#       • recommendation ranking
#       • auto-bet execution (optional)
#       • dashboard export
#       • alert export
# ============================================================

import pandas as pd
from loguru import logger

from src.betting.value_bets import build_value_bets
from src.betting.recommend_bets import recommend_bets
from src.betting.auto_bet import execute_bets


class RecommendationPipeline:
    """
    Unified pipeline for generating and optionally executing
    betting recommendations.

    Parameters
    ----------
    bankroll : float
        Current bankroll used for stake sizing.
    min_edge : float | None
        Minimum required edge for a bet to be considered.
    min_ev : float | None
        Minimum expected value per unit.
    max_kelly : float | None
        Maximum Kelly fraction allowed.
    """

    def __init__(
        self,
        bankroll: float,
        min_edge: float | None = None,
        min_ev: float | None = None,
        max_kelly: float | None = None,
    ):
        self.bankroll = bankroll
        self.min_edge = min_edge
        self.min_ev = min_ev
        self.max_kelly = max_kelly

    # --------------------------------------------------------
    # Main pipeline
    # --------------------------------------------------------
    def run(
        self,
        predictions: pd.DataFrame,
        odds: pd.DataFrame,
        execute: bool = False,
        dry_run: bool = True,
    ) -> pd.DataFrame:
        """
        Run the unified recommendation pipeline.

        Steps:
            1. Build value bets
            2. Rank + filter recommendations
            3. Optionally execute bets
        """
        logger.info("🏀 Starting unified recommendation pipeline")

        # Step 1 — Value bets
        value_bets = build_value_bets(predictions, odds)
        if value_bets.empty:
            logger.warning("No value bets found.")
            return value_bets

        # Step 2 — Recommendations
        recs = recommend_bets(
            value_bets,
            bankroll=self.bankroll,
            min_edge=self.min_edge,
            min_ev=self.min_ev,
            max_kelly=self.max_kelly,
        )

        if recs.empty:
            logger.warning("No recommendations after filtering.")
            return recs

        # Step 3 — Optional execution
        if execute:
            return execute_bets(recs, bankroll=self.bankroll, dry_run=dry_run)

        return recs

    # --------------------------------------------------------
    # Dashboard formatting
    # --------------------------------------------------------
    @staticmethod
    def to_dashboard(df: pd.DataFrame) -> pd.DataFrame:
        """
        Format recommendations for dashboard display.
        """
        if df.empty:
            return df

        out = df.copy()
        out["edge_pct"] = (out["edge"] * 100).round(1)
        out["win_prob_pct"] = (out["model_prob"] * 100).round(1)
        out["stake"] = out["recommended_stake"].round(2)

        return out[
            [
                "game_id",
                "market_team",
                "market_side",
                "ml",
                "stake",
                "edge_pct",
                "win_prob_pct",
                "confidence",
                "prediction_date",
                "model_version",
            ]
        ]

    # --------------------------------------------------------
    # Alert formatting
    # --------------------------------------------------------
    @staticmethod
    def to_alert(df: pd.DataFrame) -> list[str]:
        """
        Convert recommendations into human-readable alert messages.
        """
        messages: list[str] = []

        for _, row in df.iterrows():
            msg = (
                f"🏀 Bet {row['market_team']} ({row['market_side']}) ML {row['ml']}\n"
                f"• Win Prob: {row['model_prob']:.2f}\n"
                f"• Edge: {row['edge']:.3f}\n"
                f"• Stake: {row['recommended_stake']:.2f}\n"
                f"• Confidence: {row['confidence']:.1f}"
            )
            messages.append(msg)

        return messages