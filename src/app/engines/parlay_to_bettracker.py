from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Parlay → Bet Tracker Integration
# File: src/app/engines/parlay_to_bettracker.py
# Purpose: Log parlay bets into the unified bet tracker,
#          using EV-based confidence tiers.
# ============================================================

from datetime import date
from typing import List, Optional

from src.app.engines.bet_tracker import BetRecord, append_bet, new_bet_id
from src.app.engines.parlay import ParlayLeg
from src.app.engines.betting_math import decimal_to_american


def _confidence_from_ev_ratio(ev_ratio: float) -> str:
    # EV ratio = EV / stake (e.g. 0.05 = +5% long-term edge)
    if ev_ratio >= 0.10:
        return "High"
    if ev_ratio >= 0.05:
        return "Medium"
    if ev_ratio >= 0.02:
        return "Low"
    return "None"


def log_parlay(
    legs: List[ParlayLeg],
    stake: float,
    win_prob: float,
    decimal_odds: float,
    ev: float,
    source: Optional[str] = "parlay_builder",
    model_version: Optional[str] = None,
) -> BetRecord:
    if not legs:
        raise ValueError("Cannot log parlay with zero legs.")

    decimal_odds = max(decimal_odds, 1.0001)
    american_odds = decimal_to_american(decimal_odds)

    ev_ratio = ev / stake if stake > 0 else 0.0
    confidence = _confidence_from_ev_ratio(ev_ratio)

    description = "Parlay: " + " | ".join([leg.description for leg in legs])

    record = BetRecord(
        bet_id=new_bet_id(),
        date=str(date.today()),
        game_date="",  # parlays span multiple games
        market="parlay",
        team="MULTI",
        opponent="MULTI",
        bet_description=description,
        odds=float(american_odds),
        stake=float(stake),
        result="pending",
        payout=0.0,
        edge=float(ev_ratio),
        confidence=confidence,
        confidence_rank=None,
        model_version=model_version,
        source=source,
    )

    append_bet(record)
    return record
