# ============================================================
# File: src/ranking/bet.py
# Purpose: Generate betting recommendations
# Author: Your Name
# ============================================================

import logging


def generate_betting_recommendations(pred_df, config):
    """Generate simple betting recommendations based on predicted win probability."""
    logger = logging.getLogger(__name__)
    if pred_df.empty:
        logger.warning("No predictions to generate betting recommendations.")
        return
    threshold = getattr(config.betting, "threshold", 0.6)
    recommendations = pred_df[pred_df["predicted_win"] >= threshold]
    logger.info(
        f"Betting recommendations: {len(recommendations)} games over threshold {threshold}"
    )
