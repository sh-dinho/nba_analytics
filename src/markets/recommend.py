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
from loguru import logger

from src.analytics.team_stability import TeamStabilityEngine, TeamStabilityConfig


# ------------------------------------------------------------
# Confidence scoring
# ------------------------------------------------------------
def compute_confidence(edge, volatility_penalty, stability_score):
    """
    Produces a 0–100 confidence score.
    """
    base = min(max(edge, 0), 20) / 20  # normalize edge to [0,1]
    stability = stability_score / 100
    penalty = max(0, 1 - volatility_penalty)

    score = base * 0.5 + stability * 0.4 + penalty * 0.1
    return round(score * 100, 1)


# ------------------------------------------------------------
# Risk flags
# ------------------------------------------------------------
def compute_risk_flags(team, stability_df):
    """
    Returns a list of risk flags for a team.
    """
    row = stability_df[stability_df["team"] == team]
    if row.empty:
        return []

    row = row.iloc[0]
    flags = []

    if row["stability_score"] < 40:
        flags.append("Low stability")

    if row["roi"] < -0.05:
        flags.append("Negative ROI")

    if row["volatility"] > stability_df["volatility"].median():
        flags.append("High volatility")

    return flags


# ------------------------------------------------------------
# Main recommendation engine
# ------------------------------------------------------------
def generate_recommendations(
    ml_df: pd.DataFrame,
    totals_df: pd.DataFrame,
    spread_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    start_date=None,
    end_date=None,
):
    """
    Produces a unified recommendation table across ML + O/U + ATS.
    """

    logger.info("🏀 Generating automated betting recommendations")

    # --------------------------------------------------------
    # Load team stability metrics
    # --------------------------------------------------------
    stab_cfg = TeamStabilityConfig(
        start_date=start_date,
        end_date=end_date,
        min_bets=20,
    )
    stab_engine = TeamStabilityEngine(stab_cfg)
    stab_res = stab_engine.run()
    stability_df = stab_res.teams

    recs = []

    # --------------------------------------------------------
    # Moneyline recommendations
    # --------------------------------------------------------
    if not ml_df.empty:
        for _, r in ml_df.iterrows():
            team = r["team"]
            opp = r["opponent"]
            wp = r["win_probability"]

            edge = wp - 0.5  # simple baseline
            volatility_penalty = (
                stability_df.set_index("team").loc[team]["volatility"]
                if team in stability_df["team"].values
                else 0
            )
            stability_score = (
                stability_df.set_index("team").loc[team]["stability_score"]
                if team in stability_df["team"].values
                else 50
            )

            confidence = compute_confidence(
                edge * 100, volatility_penalty, stability_score
            )
            flags = compute_risk_flags(team, stability_df)

            recs.append(
                {
                    "market": "Moneyline",
                    "game_id": r["game_id"],
                    "team": team,
                    "opponent": opp,
                    "recommendation": f"{team} ML",
                    "edge": round(edge, 3),
                    "confidence": confidence,
                    "risk_flags": ", ".join(flags) if flags else "",
                }
            )

    # --------------------------------------------------------
    # Totals recommendations
    # --------------------------------------------------------
    if not totals_df.empty and "market_total" in odds_df.columns:
        merged = totals_df.merge(
            odds_df[["game_id", "market_total"]],
            on="game_id",
            how="left",
        )

        for _, r in merged.iterrows():
            edge_over = r["predicted_total"] - r["market_total"]
            edge_under = r["market_total"] - r["predicted_total"]

            direction = "OVER" if edge_over > edge_under else "UNDER"
            edge = max(edge_over, edge_under)

            confidence = compute_confidence(edge, 0, 50)

            recs.append(
                {
                    "market": "Totals",
                    "game_id": r["game_id"],
                    "team": f"{r['home_team']} vs {r['away_team']}",
                    "opponent": "",
                    "recommendation": direction,
                    "edge": round(edge, 2),
                    "confidence": confidence,
                    "risk_flags": "",
                }
            )

    # --------------------------------------------------------
    # Spread recommendations
    # --------------------------------------------------------
    if not spread_df.empty and "market_spread" in odds_df.columns:
        merged = spread_df.merge(
            odds_df[["game_id", "market_spread"]],
            on="game_id",
            how="left",
        )

        for _, r in merged.iterrows():
            edge_home = r["predicted_margin"] - r["market_spread"]
            edge_away = r["market_spread"] - r["predicted_margin"]

            direction = "HOME ATS" if edge_home > edge_away else "AWAY ATS"
            edge = max(edge_home, edge_away)

            confidence = compute_confidence(edge, 0, 50)

            recs.append(
                {
                    "market": "Spread",
                    "game_id": r["game_id"],
                    "team": f"{r['home_team']} vs {r['away_team']}",
                    "opponent": "",
                    "recommendation": direction,
                    "edge": round(edge, 2),
                    "confidence": confidence,
                    "risk_flags": "",
                }
            )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------
    df = pd.DataFrame(recs)
    df = df.sort_values("confidence", ascending=False)

    logger.success(f"🏀 Generated {len(df)} betting recommendations")
    return df
