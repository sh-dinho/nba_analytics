from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Parlay Engine
# File: src/app/engines/parlay.py
# Purpose: Represent parlay legs, compute parlay odds, win
#          probability, and expected value.
# ============================================================

from dataclasses import dataclass
from typing import List, Optional

from src.app.engines.betting_math import american_to_decimal, decimal_to_american


@dataclass
class ParlayLeg:
    description: str
    odds: float
    win_prob: float
    game_id: Optional[str] = None
    team: Optional[str] = None
    opponent: Optional[str] = None
    market: Optional[str] = None
    source: Optional[str] = None  # e.g. "automated", "manual", "smart_parlay"


def parlay_decimal_odds(legs: List[ParlayLeg]) -> float:
    dec = 1.0
    for leg in legs:
        dec *= american_to_decimal(leg.odds)
    return dec


def parlay_win_prob(legs: List[ParlayLeg]) -> float:
    p = 1.0
    for leg in legs:
        # Clamp probabilities for safety
        wp = max(0.0001, min(leg.win_prob, 0.9999))
        p *= wp
    return p


def parlay_expected_value(legs: List[ParlayLeg], stake: float) -> dict:
    if not legs:
        return {"win_prob": 0.0, "decimal_odds": 0.0, "ev": 0.0, "american_odds": 0.0}

    dec_odds = parlay_decimal_odds(legs)
    win_p = parlay_win_prob(legs)

    profit_if_win = stake * (dec_odds - 1.0)
    ev = win_p * profit_if_win - (1 - win_p) * stake

    american_odds = decimal_to_american(dec_odds)

    return {
        "win_prob": float(win_p),
        "decimal_odds": float(dec_odds),
        "american_odds": float(american_odds),
        "ev": float(ev),
    }
