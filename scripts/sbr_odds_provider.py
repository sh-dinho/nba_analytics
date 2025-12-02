# ============================================================
# File: scripts/sbr_odds_provider.py
# Purpose: Provide sportsbook odds data
# ============================================================

import logging
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SbrOddsProvider:
    """
    Odds provider stub. In production, this would fetch real sportsbook odds
    (e.g., via API or scraping). For now, it returns dummy odds so the pipeline
    can run end-to-end without errors.
    """

    def __init__(self, sportsbook="fanduel"):
        self.sportsbook = sportsbook

    def get_odds(self):
        """
        Return odds data for today's games.
        Structure:
        {
            "HOME:AWAY": {
                "HOME": {"money_line_odds": +150},
                "AWAY": {"money_line_odds": -150},
                "under_over_odds": 210.5
            }
        }
        """
        try:
            # TODO: implement real odds fetching logic here
            # For now, generate dummy odds for testing
            teams = [("LAL", "BOS"), ("GSW", "MIA"), ("NYK", "CHI")]
            odds_data = {}

            for home, away in teams:
                odds_data[f"{home}:{away}"] = {
                    home: {"money_line_odds": random.choice([+120, +150, -110])},
                    away: {"money_line_odds": random.choice([-120, -150, +110])},
                    "under_over_odds": random.choice([208.5, 210.5, 215.0])
                }

            logger.info(f"✅ Generated dummy odds for {len(teams)} games")
            return odds_data

        except Exception as e:
            logger.error(f"❌ Failed to fetch odds: {e}")
            # Fallback single dummy game
            return {
                "LAL:BOS": {
                    "LAL": {"money_line_odds": +150},
                    "BOS": {"money_line_odds": -150},
                    "under_over_odds": 210.5
                }
            }