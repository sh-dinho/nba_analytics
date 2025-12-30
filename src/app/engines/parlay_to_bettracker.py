from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Parlay → Bet Tracker Integration
# File: src/app/engines/parlay_to_bettracker.py
# ============================================================

from datetime import date
from typing import List

from src.app.engines.bet_tracker import BetRecord, append_bet, new_bet_id
from src.app.engines.parlay import ParlayLeg, decimal_to_american


def _confidence_from_ev(ev: float) -> str:
    if ev >= 50:
        return "High"
    if ev >= 20:
        return "Medium"
    if ev >= 5:
        return "Low"
    return "None"


def log_parlay(
    legs: List[ParlayLeg],
    stake: float,
    win_prob: float,
    decimal_odds: float,
    ev: float,
) -> BetRecord:
    description = "Parlay: " + " | ".join([leg.description for leg in legs])
    american_odds = decimal_to_american(decimal_odds)

    record = BetRecord(
        bet_id=new_bet_id(),
        date=str(date.today()),
        game_date=str(date.today()),
        market="parlay",
        team="MULTI",
        opponent="MULTI",
        bet_description=description,
        odds=float(american_odds),
        stake=float(stake),
        result="pending",
        payout=0.0,
        edge=float(ev / stake) if stake > 0 else None,
        confidence=_confidence_from_ev(ev),
    )

    append_bet(record)
    return record
