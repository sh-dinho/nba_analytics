from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Best Bets Engine
# File: src/app/engines/best_bets.py
#
# Description:
#     Combines moneyline, totals, and spread predictions into a
#     single "best bets" table with:
#       - market
#       - bet description
#       - odds (if available)
#       - model win probability (if available)
#       - edge
#       - confidence tier (High/Medium/Low)
# ============================================================

from datetime import date
from typing import Optional

import pandas as pd

from src.model.predict import run_prediction_for_date
from src.model.predict_totals import run_totals_prediction_for_date
from src.model.predict_spread import run_spread_prediction_for_date


def implied_prob(odds: Optional[float]) -> Optional[float]:
    if odds is None:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


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
    ml = run_prediction_for_date(pred_date)
    tot = run_totals_prediction_for_date(pred_date)
    sp = run_spread_prediction_for_date(pred_date)

    bets = []

    # Moneyline
    if not ml.empty and {"moneyline_odds", "win_probability"} <= set(ml.columns):
        ml = ml.copy()
        ml["implied"] = ml["moneyline_odds"].apply(implied_prob)
        ml["edge"] = ml["win_probability"] - ml["implied"]

        for _, r in ml.iterrows():
            if r["edge"] <= 0:
                continue
            bets.append(
                {
                    "market": "Moneyline",
                    "team": r["team"],
                    "opponent": r["opponent"],
                    "bet": f"{r['team']} ML",
                    "odds": float(r["moneyline_odds"]),
                    "win_prob": float(r["win_probability"]),
                    "edge": float(r["edge"]),
                    "confidence": _confidence_from_edge_pct(float(r["edge"])),
                }
            )

    # Totals
    if not tot.empty and {"total_line", "predicted_total_points"} <= set(tot.columns):
        tot = tot.copy()
        tot["diff"] = tot["predicted_total_points"] - tot["total_line"]

        for _, r in tot.iterrows():
            diff = float(r["diff"])
            if abs(diff) < 2:
                continue
            direction = "Over" if diff > 0 else "Under"
            bets.append(
                {
                    "market": "Totals",
                    "team": r["home_team"],
                    "opponent": r["away_team"],
                    "bet": f"{direction} {r['total_line']}",
                    "odds": None,
                    "win_prob": None,
                    "edge": abs(diff),
                    "confidence": _confidence_from_points(abs(diff)),
                }
            )

    # Spread
    if not sp.empty and {"spread_line", "predicted_margin"} <= set(sp.columns):
        sp = sp.copy()
        sp["diff"] = sp["predicted_margin"] - sp["spread_line"]

        for _, r in sp.iterrows():
            diff = float(r["diff"])
            if abs(diff) < 2:
                continue
            direction = r["home_team"] if diff > 0 else r["away_team"]
            bets.append(
                {
                    "market": "Spread",
                    "team": r["home_team"],
                    "opponent": r["away_team"],
                    "bet": f"{direction} ATS",
                    "odds": None,
                    "win_prob": None,
                    "edge": abs(diff),
                    "confidence": _confidence_from_points(abs(diff)),
                }
            )

    df = pd.DataFrame(bets)
    if df.empty:
        return df

    return df.sort_values(["confidence", "edge"], ascending=[True, False])
