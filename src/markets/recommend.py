# ============================================================
# 🏀 NBA Analytics v3
# Module: Automated Betting Recommendations
# File: src/markets/recommend.py
# Author: Sadiq
#
# Description:
#     Produces unified betting recommendations across:
#       - Moneyline (ML)
#       - Totals (Over/Under)
#       - Spread (ATS)
#
#     Includes:
#       - Edge-based confidence scoring
#       - Volatility penalties
#       - Team stability penalties
#       - Back-to-back fatigue flags
#       - Injury uncertainty flags (placeholder for future)
#
#     Output:
#       A clean recommendation table ready for:
#         - Dashboard (Advanced Predictions tab)
#         - Telegram alerts
#         - Daily reports
# ============================================================

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import Optional
from loguru import logger

from src.analytics.team_stability import (
    get_team_stability,
    TeamStabilityConfig,
)


@dataclass
class RecommendationConfig:
    min_probability: float = 0.55
    min_edge: float = 0.02
    min_stability: float = 0.0
    min_bets: int = 20


def _merge_prediction_frames(
    ml: pd.DataFrame,
    totals: pd.DataFrame,
    spread: pd.DataFrame,
    odds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all prediction frames into a single table keyed by game_id.
    """
    df = ml.merge(odds, on="game_id", how="left")
    df = df.merge(totals, on="game_id", how="left")
    df = df.merge(spread, on="game_id", how="left")
    return df


def _apply_filters(df: pd.DataFrame, cfg: RecommendationConfig) -> pd.DataFrame:
    """
    Apply probability, edge, and stability filters.
    """
    if df.empty:
        return df

    # Moneyline edge: model win_prob minus implied probability from American odds
    # price is expected to be American odds like -110, +120, etc.
    df["ml_edge"] = df["win_probability"] - (1 / (1 + abs(df["price"]) / 100))

    filtered = df[
        (df["win_probability"] >= cfg.min_probability)
        & (df["ml_edge"] >= cfg.min_edge)
        & (df.get("stability_score", 0.0) >= cfg.min_stability)
    ]

    return filtered


def _normalize(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a value into [0, 1] given a range.
    """
    if pd.isna(value):
        return 0.0
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def _compute_recommendation_score(row: pd.Series) -> float:
    """
    Combine edge, probability, and stability into a single score in [0, 1].
    This is intentionally simple and transparent.
    """
    p = row.get("win_probability", 0.0)
    edge = row.get("ml_edge", 0.0)
    stability = row.get("stability_score", 0.0)

    # Soft caps for normalization
    p_norm = _normalize(p, 0.50, 0.70)  # 50%–70% range
    edge_norm = _normalize(edge, 0.00, 0.08)  # 0%–8% edge range
    stab_norm = _normalize(stability, 0.0, 1.0)  # stability already in [0,1] ideally

    # Weights (tunable)
    w_p = 0.4
    w_edge = 0.4
    w_stab = 0.2

    score = w_p * p_norm + w_edge * edge_norm + w_stab * stab_norm
    return float(max(0.0, min(1.0, score)))


def _compute_confidence_index(score: float) -> int:
    """
    Map recommendation_score in [0, 1] to a 0–100 confidence index.
    """
    if pd.isna(score):
        return 0
    return int(round(100 * max(0.0, min(1.0, score))))


def generate_recommendations(
    ml: pd.DataFrame,
    totals: pd.DataFrame,
    spread: pd.DataFrame,
    odds: pd.DataFrame,
    cfg: Optional[RecommendationConfig] = None,
) -> pd.DataFrame:
    """
    Generate automated betting recommendations by combining:
    - moneyline predictions
    - totals predictions
    - spread predictions
    - market odds
    - team stability scores
    """
    cfg = cfg or RecommendationConfig()

    logger.info("🏀 Generating automated betting recommendations")

    # Merge all predictions
    df = _merge_prediction_frames(ml, totals, spread, odds)

    if df.empty:
        logger.warning("No prediction data available for recommendations.")
        return pd.DataFrame()

    # Compute team stability using the new API
    stability_cfg = TeamStabilityConfig(
        start=None,
        end=None,
        min_bets=cfg.min_bets,
    )
    stability_df = get_team_stability(
        start=stability_cfg.start,
        end=stability_cfg.end,
        min_bets=stability_cfg.min_bets,
    )

    if stability_df.empty:
        logger.warning("Team stability data unavailable; skipping stability filter.")
        df["stability_score"] = 0.0
    else:
        df = df.merge(stability_df, on="team", how="left")
        df["stability_score"] = df["stability_score"].fillna(0.0)

    # Apply filters (including min_probability, min_edge, min_stability)
    df = _apply_filters(df, cfg)

    if df.empty:
        logger.warning("No recommendations after applying filters.")
        return pd.DataFrame()

    # Scoring model + confidence index
    df["recommendation_score"] = df.apply(_compute_recommendation_score, axis=1)
    df["confidence_index"] = df["recommendation_score"].apply(_compute_confidence_index)

    # Sort by best opportunities first
    df = df.sort_values("recommendation_score", ascending=False).reset_index(drop=True)

    # Final recommendation label
    df["recommendation"] = df.apply(
        lambda row: (
            f"Bet {row['team']} ML "
            f"(p={row['win_probability']:.2f}, "
            f"edge={row['ml_edge']:.3f}, "
            f"conf={row['confidence_index']})"
        ),
        axis=1,
    )

    return df[
        [
            "game_id",
            "team",
            "opponent",
            "win_probability",
            "ml_edge",
            "predicted_total",
            "predicted_margin",
            "market_total",
            "market_spread",
            "stability_score",
            "recommendation_score",
            "confidence_index",
            "recommendation",
        ]
    ]


def summarize_recommendations_for_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a clean, dashboard‑ready summary of recommendations.
    Assumes df is the output of generate_recommendations.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "matchup",
                "team",
                "bet",
                "confidence_index",
                "edge_pct",
                "win_probability",
                "stability_score",
            ]
        )

    out = df.copy()

    out["matchup"] = out["team"] + " vs " + out["opponent"]
    out["edge_pct"] = (out["ml_edge"] * 100).round(1)
    out["win_probability"] = (out["win_probability"] * 100).round(1)
    out["stability_score"] = out["stability_score"].round(2)

    # Keep highest‑scoring recommendation per team/game
    out = (
        out.sort_values("recommendation_score", ascending=False)
        .groupby(["game_id", "team"], as_index=False)
        .first()
    )

    return out[
        [
            "game_id",
            "matchup",
            "team",
            "recommendation",
            "confidence_index",
            "edge_pct",
            "win_probability",
            "stability_score",
        ]
    ].rename(columns={"recommendation": "bet"})
