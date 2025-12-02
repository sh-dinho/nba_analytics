from sbrscrape import Scoreboard

class SbrOddsProvider:
    """
    Provides NBA odds from SBR Scoreboard.
    Normalizes team names and extracts moneyline + totals for a given sportsbook.
    """

    TEAM_NAME_MAP = {
        "Los Angeles Clippers": "LA Clippers",
        "Los Angeles Lakers": "LA Lakers",
        # Add more mappings as needed
    }

    def __init__(self, sportsbook="fanduel"):
        sb = Scoreboard(sport="NBA")
        self.games = getattr(sb, "games", [])
        self.sportsbook = sportsbook

    def normalize_team_name(self, name: str) -> str:
        return self.TEAM_NAME_MAP.get(name, name)

    def get_odds(self):
        """
        Returns:
            dict: {
                "HomeTeam:AwayTeam": {
                    "under_over_odds": float or None,
                    "home_team": {"money_line_odds": float or None},
                    "away_team": {"money_line_odds": float or None}
                }
            }
        """
        dict_res = {}
        for game in self.games:
            home_team_name = self.normalize_team_name(game.get("home_team", "Unknown"))
            away_team_name = self.normalize_team_name(game.get("away_team", "Unknown"))

            home_ml = game.get("home_ml", {})
            away_ml = game.get("away_ml", {})
            totals = game.get("total", {})

            money_line_home_value = home_ml.get(self.sportsbook)
            money_line_away_value = away_ml.get(self.sportsbook)
            totals_value = totals.get(self.sportsbook)

            dict_res[f"{home_team_name}:{away_team_name}"] = {
                "under_over_odds": totals_value,
                home_team_name: {"money_line_odds": money_line_home_value},
                away_team_name: {"money_line_odds": money_line_away_value},
            }

        return dict_res