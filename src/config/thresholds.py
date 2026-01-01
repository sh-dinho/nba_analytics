from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Threshold Configuration
# File: src/config/thresholds.py
# Author: Sadiq
#
# Description:
#     Centralized thresholds for filtering value bets and
#     generating recommendations.
#
#     These values are used by:
#       • value_bet_engine
#       • recommend_bets
#       • dashboards & reports
# ============================================================


# ------------------------------------------------------------
# Minimum Model Edge
# ------------------------------------------------------------
# Minimum required difference between model probability and
# market implied probability to consider a bet actionable.
#
# Example:
#   model_prob = 0.58
#   implied_prob = 0.52
#   edge = 0.06  → passes threshold
#
MIN_EDGE: float = 0.02   # 2% minimum edge


# ------------------------------------------------------------
# Minimum Expected Value (EV per unit)
# ------------------------------------------------------------
# Expected value per unit staked must exceed this threshold.
#
# EV per unit = (model_prob * decimal_odds) - 1
#
MIN_EV: float = 0.01     # 1% minimum EV


# ------------------------------------------------------------
# Minimum Win Probability
# ------------------------------------------------------------
# Optional: some users prefer to filter out very low-probability
# bets even if they have high EV. This is not enforced by default
# in recommend_bets(), but available for future use.
#
MIN_WIN_PROB: float = 0.50   # 50% default cutoff (not enforced unless used)


# ------------------------------------------------------------
# Minimum Confidence Score
# ------------------------------------------------------------
# Optional: if you want to filter recommendations by confidence.
# Not used by default, but available for dashboards or alerts.
#
MIN_CONFIDENCE: float = 0.0   # 0–100 scale