# File: nba_analytics_core/team_strength.py

import pandas as pd
import numpy as np
import random

def update_elo(games, k=20, base=1500, home_adv=65):
    """
    Compute team Elo ratings from chronological games list.
    Tracks Elo history for each game.
    
    Parameters:
        games: list of dict {game_id, home_team, away_team, home_win, importance(optional)}
        k: base K-factor
        base: starting Elo rating
        home_adv: Elo points added to home team
    
    Returns:
        elo: dict team -> final Elo rating
        history: DataFrame of Elo ratings after each game
    """
    elo = {}
    history = []

    def get(team): return elo.get(team, base)

    for g in games:
        h, a = g["home_team"], g["away_team"]
        eh, ea = get(h), get(a)

        # Expected score with home advantage
        exp_h = 1 / (1 + 10 ** ((ea - (eh + home_adv)) / 400))
        score_h = 1 if g["home_win"] else 0

        # Dynamic K-factor (importance scaling)
        k_factor = g.get("importance", k)

        elo[h] = eh + k_factor * (score_h - exp_h)
        elo[a] = ea + k_factor * ((1 - score_h) - (1 - exp_h))

        history.append({
            "game_id": g.get("game_id"),
            "home_team": h,
            "away_team": a,
            "home_win": g["home_win"],
            "elo_home": elo[h],
            "elo_away": elo[a]
        })

    return elo, pd.DataFrame(history)


def championship_probabilities(elo, method="softmax", sims=1000):
    """
    Convert Elo ratings to championship probabilities.
    
    Parameters:
        elo: dict team -> Elo rating
        method: "softmax" (approximation) or "simulation" (Monte Carlo playoffs)
        sims: number of simulations if method="simulation"
    
    Returns:
        DataFrame with team, Elo rating, and championship probability
    """
    teams = list(elo.keys())
    ratings = np.array([elo[t] for t in teams], dtype=float)

    if method == "softmax":
        # Temperature scaling softmax
        exps = np.exp((ratings - ratings.mean()) / 40.0)
        probs = exps / exps.sum()
        return pd.DataFrame({
            "team": teams,
            "elo": ratings,
            "championship_prob": probs
        }).sort_values("championship_prob", ascending=False)

    elif method == "simulation":
        # Monte Carlo playoff simulation (simplified bracket)
        wins = {t: 0 for t in teams}

        def win_prob(team_a, team_b):
            ea, eb = elo[team_a], elo[team_b]
            return 1 / (1 + 10 ** ((eb - ea) / 400))

        for _ in range(sims):
            alive = teams[:]
            while len(alive) > 1:
                random.shuffle(alive)
                next_round = []
                for i in range(0, len(alive), 2):
                    if i+1 >= len(alive):
                        next_round.append(alive[i])
                        continue
                    a, b = alive[i], alive[i+1]
                    if random.random() < win_prob(a, b):
                        next_round.append(a)
                    else:
                        next_round.append(b)
                alive = next_round
            wins[alive[0]] += 1

        probs = np.array([wins[t] / sims for t in teams])
        return pd.DataFrame({
            "team": teams,
            "elo": ratings,
            "championship_prob": probs
        }).sort_values("championship_prob", ascending=False)