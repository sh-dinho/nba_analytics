from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Betting Configuration
# ============================================================

BANKROLL: float = 1000.0

KELLY_CAP: float = 0.05
KELLY_MIN: float = 0.001

MAX_TOTAL_EXPOSURE: float = 0.20
MAX_EXPOSURE_PER_GAME: float = 0.10
MAX_BETS_PER_DAY: int = 5

CONFIDENCE_WEIGHTS = {
    "edge": 0.5,
    "ev": 0.3,
    "kelly": 0.2,
}