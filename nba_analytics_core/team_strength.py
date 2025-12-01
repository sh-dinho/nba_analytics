# Path: nba_analytics_core/team_strength.py
import pandas as pd
import numpy as np

def update_elo(games, k=20, base=1500):
    """
    Compute simple team Elo from chronological games list.
    games: list of dict {home_team, away_team, home_win}
    Returns dict team -> elo rating.
    """
    elo = {}
    def get(team): return elo.get(team, base)
    for g in games:
        h, a = g["home_team"], g["away_team"]
        eh, ea = get(h), get(a)
        exp_h = 1 / (1 + 10 ** ((ea - eh) / 400))
        score_h = 1 if g["home_win"] else 0
        elo[h] = eh + k * (score_h - exp_h)
        elo[a] = ea + k * ((1 - score_h) - (1 - exp_h))
    return elo

def championship_probabilities(elo):
    """
    Convert Elo ratings to normalized championship probabilities (softmax).
    """
    teams = list(elo.keys())
    ratings = np.array([elo[t] for t in teams], dtype=float)
    exps = np.exp((ratings - ratings.mean()) / 40.0)  # temperature scaling
    probs = exps / exps.sum()
    return pd.DataFrame({"team": teams, "championship_prob": probs}).sort_values("championship_prob", ascending=False)