# scripts/fetch_games_odds.py
import pandas as pd
from datetime import date
from typing import List, Tuple

def fetch_todays_games() -> List[Tuple[str, str]]:
    """
    Simulate fetching today's NBA games (replace with real API call)
    Returns list of (home_team, away_team)
    """
    # Example simulation
    return [
        ("Lakers", "Warriors"),
        ("Bulls", "Celtics"),
        ("Nets", "Heat")
    ]

def fetch_todays_odds(games: List[Tuple[str, str]]) -> Tuple[List[float], List[float], List[float]]:
    """
    Simulate fetching odds and OU lines.
    Returns home_odds, away_odds, ou_lines
    """
    # In production: fetch from Odds API or bookmaker CSV
    n = len(games)
    home_odds = [1.95 + 0.05*i for i in range(n)]
    away_odds = [1.85 + 0.05*i for i in range(n)]
    ou_lines = [220.0 + i for i in range(n)]
    return home_odds, away_odds, ou_lines
