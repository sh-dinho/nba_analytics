# ============================================================
# File: scripts/sbr_odds_provider.py
# Purpose: Provide sportsbook odds data
# ============================================================

import random
from core.log_config import setup_logger
from core.exceptions import PipelineError

logger = setup_logger("sbr_odds_provider")


class SbrOddsProvider:
    """
    Odds provider stub. In production, this would fetch real sportsbook odds
    (e.g., via API or scraping). For now, it returns dummy odds so the pipeline
    can run end-to-end without errors.
    """

    def __init__(self, sportsbook: str = "fanduel", seed: int | None = None):
        self.sportsbook = sportsbook
        if seed is not None:
            random.seed(seed)  # deterministic odds for testing

    def get_odds(self, teams: list[tuple[str, str]] | None = None) -> dict:
        """
        Return odds data for today's games.

        Args:
            teams: Optional list of (home, away) team codes. Defaults to sample teams.

        Returns:
            dict: Odds data structured as:
            {
                "HOME:AWAY": {
                    "HOME": {"money_line_odds": +150},
                    "AWAY": {"money_line_odds": -150},
                    "under_over_odds": 210.5
                }
            }
        """
        try:
            if teams is None:
                teams = [("LAL", "BOS"), ("GSW", "MIA"), ("NYK", "CHI")]

            odds_data = {}

            for home, away in teams:
                # Generate complementary odds (home positive, away negative)
                home_odds = random.choice([+120, +150, -110])
                away_odds = -home_odds if home_odds > 0 else abs(home_odds)

                odds_data[f"{home}:{away}"] = {
                    home: {"money_line_odds": home_odds},
                    away: {"money_line_odds": away_odds},
                    "under_over_odds": random.choice([208.5, 210.5, 215.0]),
                }

            logger.info(f"✅ Generated dummy odds for {len(teams)} games (sportsbook={self.sportsbook})")
            return odds_data

        except Exception as e:
            logger.error(f"❌ Failed to fetch odds: {e}")
            raise PipelineError("Odds provider failed to generate odds.")