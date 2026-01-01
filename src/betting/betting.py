from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Betting Configuration
# File: src/config/betting.py
# Author: Sadiq
#
# Description:
#     Centralized configuration for bankroll management,
#     Kelly sizing, exposure limits, and confidence scoring.
#
#     Used by:
#       • value_bet_engine
#       • recommend_bets
#       • auto_bet executor
#       • dashboards & reports
# ============================================================


# ------------------------------------------------------------
# Bankroll
# ------------------------------------------------------------
# Default bankroll used for sizing bets (can be overridden)
BANKROLL: float = 1000.0


# ------------------------------------------------------------
# Kelly Sizing
# ------------------------------------------------------------
# Maximum allowed Kelly fraction per bet (safety cap)
KELLY_CAP: float = 0.05     # 5% of bankroll max

# Minimum Kelly fraction (prevents rounding to zero)
KELLY_MIN: float = 0.001    # 0.1% of bankroll


# ------------------------------------------------------------
# Exposure Limits
# ------------------------------------------------------------
# Maximum total exposure across all bets for a given day
MAX_TOTAL_EXPOSURE: float = 0.20   # 20% of bankroll

# Maximum exposure allowed on a single game
MAX_EXPOSURE_PER_GAME: float = 0.10  # 10% of bankroll

# Maximum number of bets allowed per day
MAX_BETS_PER_DAY: int = 5


# ------------------------------------------------------------
# Confidence Scoring Weights
# ------------------------------------------------------------
# These weights determine how the recommendation engine
# blends edge, EV, and Kelly into a 0–100 confidence score.
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "edge": 0.5,     # model edge importance
    "ev": 0.3,       # expected value importance
    "kelly": 0.2,    # Kelly fraction importance
}


# ------------------------------------------------------------
# Validation (executed at import time)
# ------------------------------------------------------------
def _validate_config() -> None:
    # Kelly caps
    if not (0 <= KELLY_MIN <= KELLY_CAP <= 1):
        raise ValueError("Invalid Kelly configuration: ensure 0 ≤ KELLY_MIN ≤ KELLY_CAP ≤ 1")

    # Exposure limits
    if not (0 <= MAX_EXPOSURE_PER_GAME <= MAX_TOTAL_EXPOSURE <= 1):
        raise ValueError("Exposure limits must satisfy: per-game ≤ total ≤ 1")

    # Confidence weights
    total_weight = sum(CONFIDENCE_WEIGHTS.values())
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"CONFIDENCE_WEIGHTS must sum to 1.0 (got {total_weight})")


_validate_config()