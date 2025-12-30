from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Automated Betting Recommendations
# File: src/markets/recommend.py
# ============================================================

from dataclasses import dataclass
from typing import Optional

import pandas as pd
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


def implied_prob(odds: float) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def _normalize(value: float, min_val: float, max_val: float) -> float:
    if pd.isna(value) or max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def _compute_recommendation_score(row: pd.Series) -> float:
    p = row.get("win_probability", 0.0)
    edge = row.get("ml_edge", 0.0)
    stability = row.get("stability_score", 0.0)

    p_norm = _normalize(p, 0.50, 0.70)
    edge_norm = _normalize(edge, 0.00, 0.08)
    stab_norm = _normalize(stability, 0.0, 1.0)

    w_p = 0.4
    w_edge = 0.4
    w_stab = 0.2

    score = w_p * p_norm + w_edge * edge_norm + w_stab * stab_norm
    return float(max(0.0, min(1.0, score)))


def _compute_confidence_index(score: float) -> int:
    if pd.isna(score):
        return 0
    return int(round(100 * max(0.0, min(1.0, score))))


def _merge_prediction_frames(
    ml: pd.DataFrame,
    totals: pd.DataFrame,
    spread: pd.DataFrame,
    odds: pd.DataFrame,
) -> pd.DataFrame:
    df = ml.merge(odds, on="game_id", how="left")
    df = df.merge(totals, on="game_id", how="left")
    df = df.merge(spread, on="game_id", how="left")
    return df


def _apply_filters(df: pd.DataFrame, cfg: RecommendationConfig) -> pd.DataFrame:
    if df.empty:
        return df

    if "price" not in df.columns:
        df["ml_edge"] = 0.0
    else:
        df["ml_edge"] = df.apply(
            lambda r: (
                r["win_probability"] - implied_prob(r["price"])
                if pd.notna(r.get("price"))
                else 0.0
            ),
            axis=1,
        )

    filtered = df[
        (df["win_probability"] >= cfg.min_probability)
        & (df["ml_edge"] >= cfg.min_edge)
        & (df.get("stability_score", 0.0) >= cfg.min_stability)
    ]

    return filtered


def generate_recommendations(
    ml: pd.DataFrame,
    totals: pd.DataFrame,
    spread: pd.DataFrame,
    odds: pd.DataFrame,
    cfg: Optional[RecommendationConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or RecommendationConfig()

    logger.info("🏀 Generating automated betting recommendations (v4)")

    df = _merge_prediction_frames(ml, totals, spread, odds)

    if df.empty:
        logger.warning("No prediction data available for recommendations.")
        return pd.DataFrame()

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
        logger.warning("Team stability data unavailable; using zero stability.")
        df["stability_score"] = 0.0
    else:
        df = df.merge(stability_df, on="team", how="left")
        df["stability_score"] = df["stability_score"].fillna(0.0)

    df = _apply_filters(df, cfg)

    if df.empty:
        logger.warning("No recommendations after applying filters.")
        return pd.DataFrame()

    df["recommendation_score"] = df.apply(_compute_recommendation_score, axis=1)
    df["confidence_index"] = df["recommendation_score"].apply(_compute_confidence_index)

    for col in ["predicted_total", "market_total", "predicted_margin", "market_spread"]:
        if col not in df.columns:
            df[col] = pd.NA

    df["recommendation"] = df.apply(
        lambda row: (
            f"Bet {row['team']} ML "
            f"(p={row['win_probability']:.2f}, "
            f"edge={row['ml_edge']:.3f}, "
            f"conf={row['confidence_index']})"
        ),
        axis=1,
    )

    df = df.sort_values("recommendation_score", ascending=False).reset_index(drop=True)

    return df[
        [
            "game_id",
            "team",
            "opponent",
            "win_probability",
            "price",
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
                "price",
            ]
        )

    out = df.copy()

    out["matchup"] = out["team"] + " vs " + out["opponent"]
    out["edge_pct"] = (out["ml_edge"] * 100).round(1)
    out["win_probability"] = (out["win_probability"] * 100).round(1)
    out["stability_score"] = out["stability_score"].round(2)

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
            "price",
        ]
    ].rename(columns={"recommendation": "bet"})
