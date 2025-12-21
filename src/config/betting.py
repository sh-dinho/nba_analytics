# ============================================================
# Project: NBA Analytics & Betting Engine
# Author: Sadiq
# Description: Bankroll, Kelly, and exposure rules for bet sizing.
# ============================================================

BANKROLL = 100

KELLY_CAP = 0.05
KELLY_MIN = 0.0

MAX_TOTAL_EXPOSURE = 0.20
MAX_EXPOSURE_PER_GAME = 0.05
MAX_BETS_PER_DAY = 5

CONFIDENCE_WEIGHTS = {
    "edge": 0.50,
    "ev": 0.30,
    "kelly": 0.20,
}
