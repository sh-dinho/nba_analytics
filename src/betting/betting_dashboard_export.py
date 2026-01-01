from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Dashboard Export
# File: src/betting/betting_dashboard_export.py
# Author: Sadiq
#
# Description:
#     Export recommendations, executed bets, and bankroll curve
#     into dashboard‑friendly JSON files.
# ============================================================

import pandas as pd
from loguru import logger

from src.config.paths import (
    DASHBOARD_DIR,
    DASHBOARD_RECOMMENDATIONS_PATH,
    DASHBOARD_BANKROLL_PATH,
)


# ------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------
def _ensure_dashboard_dir() -> None:
    """Ensure the dashboard directory exists."""
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Export: Recommendations
# ------------------------------------------------------------
def export_recommendations(df: pd.DataFrame) -> None:
    """
    Export recommended bets to a dashboard‑friendly JSON file.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the recommendation fields produced by
        the recommendation engine.
    """
    if df is None or df.empty:
        logger.warning("Dashboard export: recommendations DataFrame is empty.")
        return

    _ensure_dashboard_dir()
    df.to_json(DASHBOARD_RECOMMENDATIONS_PATH, orient="records", indent=2)

    logger.success(
        f"Dashboard export: recommendations → {DASHBOARD_RECOMMENDATIONS_PATH}"
    )


# ------------------------------------------------------------
# Export: Bankroll Curve
# ------------------------------------------------------------
def export_bankroll_curve(curve: pd.DataFrame) -> None:
    """
    Export bankroll curve to a dashboard‑friendly JSON file.

    Parameters
    ----------
    curve : pd.DataFrame
        Must contain bankroll history fields:
        timestamp, bankroll, profit, etc.
    """
    if curve is None or curve.empty:
        logger.warning("Dashboard export: bankroll curve DataFrame is empty.")
        return

    _ensure_dashboard_dir()
    curve.to_json(DASHBOARD_BANKROLL_PATH, orient="records", indent=2)

    logger.success(
        f"Dashboard export: bankroll curve → {DASHBOARD_BANKROLL_PATH}"
    )


# ------------------------------------------------------------
# Export: Combined
# ------------------------------------------------------------
def export_all(
    recommendations: pd.DataFrame,
    bankroll_curve: pd.DataFrame,
) -> None:
    """
    Export both recommendations and bankroll curve.

    Parameters
    ----------
    recommendations : pd.DataFrame
    bankroll_curve : pd.DataFrame
    """
    try:
        export_recommendations(recommendations)
    except Exception as e:
        logger.error(f"Failed to export recommendations: {e}")

    try:
        export_bankroll_curve(bankroll_curve)
    except Exception as e:
        logger.error(f"Failed to export bankroll curve: {e}")