# ============================================================
# File: scripts/sbr_odds_provider.py
# Purpose: Provide sportsbook odds data (stub for testing)
# ============================================================

import random
import argparse
import json
from core.log_config import init_global_logger
from core.exceptions import PipelineError

logger = init_global_logger()


class SbrOddsProvider:
    """
    Odds provider stub. In production, this would fetch real sportsbook odds
    (e.g., via API or scraping). For now, it returns dummy odds so the pipeline
    can run end-to-end without errors.
    """

    def __init__(self, sportsbook: str = "fanduel", seed: int | None = None):
        self.sportsbook = sportsbook
        self._rng = random.Random(seed) if seed is not None else random.Random()

    def get_odds(self, teams: list[tuple[str, str]] | None = None,
                 num_games: int = 3,
                 under_over_range: list[float] = [208.5, 210.5, 215.0]) -> dict:
        """
        Return odds data for today's games.

        Args:
            teams: Optional list of (home, away) team codes. Defaults to sample teams.
            num_games: Number of dummy games to generate if teams not provided.
            under_over_range: List of possible totals for under/over odds.

        Returns:
            dict: Odds data structured as:
            {
                "HOME:AWAY": {
                    "HOME": {"american_odds": +150},
                    "AWAY": {"american_odds": -150},
                    "under_over_total": 210.5
                }
            }
        """
        try:
            if teams is None:
                teams = [("LAL", "BOS"), ("GSW", "MIA"), ("NYK", "CHI")][:num_games]

            odds_data = {}

            for home, away in teams:
                home_odds = self._rng.choice([+120, +150, -110])
                away_odds = -home_odds if home_odds > 0 else abs(home_odds)

                odds_data[f"{home}:{away}"] = {
                    home: {"american_odds": home_odds},
                    away: {"american_odds": away_odds},
                    "under_over_total": self._rng.choice(under_over_range),
                }

            logger.info(f"✅ Generated dummy odds for {len(teams)} games (sportsbook={self.sportsbook})")
            return odds_data

        except Exception as e:
            logger.error(f"❌ Failed to fetch odds: {e}")
            raise PipelineError(f"Odds provider failed to generate odds: {e}")


# === CLI Wrapper ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy sportsbook odds")
    parser.add_argument("--sportsbook", type=str, default="fanduel", help="Sportsbook name")
    parser.add_argument("--seed", type=int, help="Random seed for deterministic odds")
    parser.add_argument("--num-games", type=int, default=3, help="Number of games to generate")
    parser.add_argument("--teams", nargs="+", help="Optional list of teams as HOME:AWAY pairs (e.g. LAL:BOS GSW:MIA)")
    args = parser.parse_args()

    provider = SbrOddsProvider(sportsbook=args.sportsbook, seed=args.seed)

    teams = None
    if args.teams:
        teams = [tuple(pair.split(":")) for pair in args.teams]

    odds = provider.get_odds(teams=teams, num_games=args.num_games)

    # Pretty-print JSON
    print(json.dumps(odds, indent=2))
