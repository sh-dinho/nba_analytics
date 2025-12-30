from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Parlay Engine
# File: src/app/engines/parlay.py
# ============================================================

from dataclasses import dataclass
from typing import List


@dataclass
class ParlayLeg:
    description: str
    odds: float
    win_prob: float


def american_to_decimal(odds: float) -> float:
    if odds > 0:
        return 1 + odds / 100.0
    return 1 + 100.0 / abs(odds)


def decimal_to_american(decimal_odds: float) -> float:
    if decimal_odds >= 2.0:
        return (decimal_odds - 1.0) * 100.0
    return -100.0 / (decimal_odds - 1.0)


def parlay_decimal_odds(legs: List[ParlayLeg]) -> float:
    dec = 1.0
    for leg in legs:
        dec *= american_to_decimal(leg.odds)
    return dec


def parlay_win_prob(legs: List[ParlayLeg]) -> float:
    p = 1.0
    for leg in legs:
        p *= leg.win_prob
    return p


def parlay_expected_value(legs: List[ParlayLeg], stake: float) -> dict:
    if not legs:
        return {"win_prob": 0.0, "decimal_odds": 0.0, "ev": 0.0}

    dec_odds = parlay_decimal_odds(legs)
    win_p = parlay_win_prob(legs)

    profit_if_win = stake * (dec_odds - 1.0)
    ev = win_p * profit_if_win - (1 - win_p) * stake

    return {
        "win_prob": float(win_p),
        "decimal_odds": float(dec_odds),
        "ev": float(ev),
    }
