from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Best Bets Engine
# File: src/app/engines/best_bets.py
# Purpose: Combine ML, totals, and spread predictions into a
#          unified best-bets table with edges and confidence.
# ============================================================

from datetime import date
from typing import Optional, Dict

import pandas as pd

from src.model.predict import run_prediction_for_date
from src.model.predict_totals import run_totals_prediction_for_date
from src.model.predict_spread import run_spread_prediction_for_date
from src.app.engines.betting_math import implied_prob

CONF_RANK: Dict[str, int] = {"High": 3, "Medium": 2, "Low": 1, "None": 0}


def _confidence_from_edge_pct(edge: float) -> str:
    if edge >= 0.07:
        return "High"
    if edge >= 0.04:
        return "Medium"
    if edge >= 0.02:
        return "Low"
    return "None"


def _confidence_from_points(diff: float) -> str:
    if diff >= 6:
        return "High"
    if diff >= 4:
        return "Medium"
    if diff >= 2:
        return "Low"
    return "None"


def compute_best_bets(pred_date: date) -> pd.DataFrame:
    """
    Build a unified best-bets table for a given prediction date.
    Moneyline bets use EV edge; totals and spreads use point-difference.
    """
    ml = run_prediction_for_date(pred_date)
    tot = run_totals_prediction_for_date(pred_date)
    sp = run_spread_prediction_for_date(pred_date)

    bets = []

    # -------------------------
    # Moneyline
    # -------------------------
    if not ml.empty and {"moneyline_odds", "win_probability"} <= set(ml.columns):
        ml = ml.copy()

        # Normalize win_probability to 0–1 if needed
        ml["win_probability"] = ml["win_probability"].apply(
            lambda x: x / 100.0 if x > 1 else x
        )

        ml["implied"] = ml["moneyline_odds"].apply(implied_prob)
        ml["edge"] = ml["win_probability"] - ml["implied"]

        for _, r in ml.iterrows():
            if r["edge"] <= 0:
                continue

            edge_val = float(r["edge"])
            conf = _confidence_from_edge_pct(edge_val)

            bets.append(
                {
                    "market": "Moneyline",
                    "team": r["team"],
                    "opponent": r["opponent"],
                    "bet": f"{r['team']} ML",
                    "odds": float(r["moneyline_odds"]),
                    "win_prob": float(r["win_probability"]),
                    "edge": edge_val,
                    "confidence": conf,
                    "confidence_rank": CONF_RANK[conf],
                }
            )

    # -------------------------
    # Totals
    # -------------------------
    if not tot.empty and {"total_line", "predicted_total_points"} <= set(tot.columns):
        tot = tot.copy()
        tot["diff"] = tot["predicted_total_points"] - tot["total_line"]

        for _, r in tot.iterrows():
            diff = float(r["diff"])
            if abs(diff) < 2:
                continue

            direction = "Over" if diff > 0 else "Under"
            diff_abs = abs(diff)
            conf = _confidence_from_points(diff_abs)

            bets.append(
                {
                    "market": "Totals",
                    "team": r["home_team"],
                    "opponent": r["away_team"],
                    "bet": f"{direction} {float(r['total_line']):.1f}",
                    "odds": None,
                    "win_prob": None,
                    "edge": diff_abs,
                    "confidence": conf,
                    "confidence_rank": CONF_RANK[conf],
                }
            )

    # -------------------------
    # Spread
    # -------------------------
    if not sp.empty and {"spread_line", "predicted_margin"} <= set(sp.columns):
        sp = sp.copy()
        sp["diff"] = sp["predicted_margin"] - sp["spread_line"]

        for _, r in sp.iterrows():
            diff = float(r["diff"])
            if abs(diff) < 2:
                continue

            line = float(r["spread_line"])
            direction_team = r["home_team"] if diff > 0 else r["away_team"]
            diff_abs = abs(diff)
            conf = _confidence_from_points(diff_abs)

            bets.append(
                {
                    "market": "Spread",
                    "team": r["home_team"],
                    "opponent": r["away_team"],
                    "bet": f"{direction_team} {line:+.1f}",
                    "odds": None,
                    "win_prob": None,
                    "edge": diff_abs,
                    "confidence": conf,
                    "confidence_rank": CONF_RANK[conf],
                }
            )

    df = pd.DataFrame(bets)
    if df.empty:
        return df

    return df.sort_values(
        ["confidence_rank", "edge"], ascending=[False, False]
    ).reset_index(drop=True)
