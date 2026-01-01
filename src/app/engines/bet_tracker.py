from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Bet Tracker Engine
# File: src/app/engines/bet_tracker.py
# Purpose: Persistent bet logging, ROI/Sharpe, Kelly, and
#          bankroll simulation utilities.
# ============================================================

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config.paths import BET_LOG_PATH

try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False


@dataclass
class BetRecord:
    bet_id: str
    date: str
    game_date: str
    market: str
    team: str
    opponent: str
    bet_description: str
    odds: float
    stake: float
    result: str
    payout: float
    edge: Optional[float] = None
    confidence: Optional[str] = None
    confidence_rank: Optional[int] = None
    model_version: Optional[str] = None
    source: Optional[str] = None  # e.g. "manual", "automated", "parlay_builder"


def _ensure_log_exists() -> None:
    BET_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not BET_LOG_PATH.exists():
        df = pd.DataFrame(
            columns=[
                "bet_id",
                "date",
                "game_date",
                "market",
                "team",
                "opponent",
                "bet_description",
                "odds",
                "stake",
                "result",
                "payout",
                "edge",
                "confidence",
                "confidence_rank",
                "model_version",
                "source",
            ]
        )
        df.to_csv(BET_LOG_PATH, index=False)
        logger.info(f"[BetTracker] Initialized bet log at {BET_LOG_PATH}")


def load_bets() -> pd.DataFrame:
    _ensure_log_exists()
    df = pd.read_csv(
        BET_LOG_PATH,
        dtype={
            "bet_id": str,
            "date": str,
            "game_date": str,
            "market": str,
            "team": str,
            "opponent": str,
            "bet_description": str,
            "odds": float,
            "stake": float,
            "result": str,
            "payout": float,
            "edge": float,
            "confidence": str,
            "confidence_rank": float,
            "model_version": str,
            "source": str,
        },
    )
    return df


def append_bet(record: BetRecord) -> None:
    _ensure_log_exists()
    new_data = pd.DataFrame([asdict(record)])

    with open(BET_LOG_PATH, "a") as f:
        try:
            if HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_EX)
            new_data.to_csv(f, header=False, index=False)
        finally:
            if HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_UN)


def _american_odds_profit(odds: float, stake: float) -> float:
    if odds > 0:
        return stake * (odds / 100.0)
    return stake * (100.0 / abs(odds))


def update_bet_result(bet_id: str, result: str) -> None:
    df = load_bets()
    if bet_id not in df["bet_id"].values:
        raise ValueError(f"Bet ID not found: {bet_id}")

    idx = df.index[df["bet_id"] == bet_id][0]
    odds = float(df.at[idx, "odds"])
    stake = float(df.at[idx, "stake"])

    if result == "win":
        profit = _american_odds_profit(odds, stake)
    elif result == "loss":
        profit = -stake
    elif result == "push":
        profit = 0.0
    else:
        raise ValueError(f"Invalid result: {result}")

    df.at[idx, "result"] = result
    df.at[idx, "payout"] = float(profit)

    with open(BET_LOG_PATH, "w") as f:
        try:
            if HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_EX)
            df.to_csv(f, index=False)
        finally:
            if HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_UN)

    logger.info(f"[BetTracker] Updated bet {bet_id} → {result}, payout={profit}")


def compute_roi(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "total_bets": 0,
            "won": 0,
            "lost": 0,
            "push": 0,
            "staked": 0.0,
            "profit": 0.0,
            "roi": 0.0,
        }

    resolved = df[df["result"].isin(["win", "loss", "push"])]
    staked = resolved["stake"].sum()
    profit = resolved["payout"].sum()

    return {
        "total_bets": len(resolved),
        "won": int((resolved["result"] == "win").sum()),
        "lost": int((resolved["result"] == "loss").sum()),
        "push": int((resolved["result"] == "push").sum()),
        "staked": float(staked),
        "profit": float(profit),
        "roi": float(profit / staked) if staked > 0 else 0.0,
    }


def compute_sharpe_ratio(df: pd.DataFrame) -> float:
    resolved = df[df["result"].isin(["win", "loss"])]
    if resolved.empty:
        return 0.0

    resolved = resolved.copy()
    resolved["return"] = resolved["payout"] / resolved["stake"]
    mu = resolved["return"].mean()
    sigma = resolved["return"].std()

    if sigma == 0:
        return 0.0

    return float(mu / sigma)


def kelly_fraction(win_prob: float, odds: float) -> float:
    """
    Kelly fraction f* = (b*p - q) / b, where:
      - b is decimal odds minus 1 for the bet
      - p is win probability
      - q = 1 - p
    """
    if win_prob > 1:
        win_prob /= 100.0
    win_prob = min(max(win_prob, 0.0001), 0.9999)

    if odds > 0:
        b = odds / 100.0
    else:
        b = 100.0 / abs(odds)

    q = 1 - win_prob
    f = (b * win_prob - q) / b
    return max(float(f), 0.0)


def simulate_bankroll(
    payouts: np.ndarray,
    initial_bankroll: float,
    n_sims: int = 3000,
    horizon: int = 200,
) -> np.ndarray:
    """
    Simple flat-stake bankroll simulation using historical payout distribution.
    """
    n = len(payouts)
    results = np.zeros((n_sims, horizon), dtype=float)

    for s in range(n_sims):
        bankroll = initial_bankroll
        for t in range(horizon):
            idx = np.random.randint(0, n)
            bankroll += payouts[idx]
            results[s, t] = bankroll

    return results


def new_bet_id() -> str:
    # High-res timestamp + random suffix to avoid collisions
    from uuid import uuid4

    return datetime.utcnow().strftime("%Y%m%d%H%M%S%f") + "_" + uuid4().hex[:6]
