# Path: nba_analytics_core/team_strength.py

import pandas as pd
import numpy as np
import random
import logging
from datetime import datetime
import os
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def update_elo(games, k=20, base=1500, home_adv=65, margin_factor=False):
    """
    Compute team Elo ratings from chronological games list.
    Tracks Elo history for each game.

    Parameters:
        games: list of dict {game_id, home_team, away_team, home_win, importance(optional), margin(optional)}
        k: base K-factor
        base: starting Elo rating
        home_adv: Elo points added to home team
        margin_factor: if True, scale updates by margin of victory

    Returns:
        elo: dict team -> final Elo rating
        history: DataFrame of Elo ratings after each game
    """
    if not games:
        logger.warning("No games provided for Elo update.")
        return {}, pd.DataFrame()

    elo = {}
    history = []

    def get(team): return elo.get(team, base)

    for g in games:
        if not {"home_team", "away_team", "home_win"}.issubset(g.keys()):
            logger.error(f"Game missing required keys: {g}")
            continue

        h, a = g["home_team"], g["away_team"]
        eh, ea = get(h), get(a)

        # Expected score with home advantage
        exp_h = 1 / (1 + 10 ** ((ea - (eh + home_adv)) / 400))
        score_h = 1 if g["home_win"] else 0

        # Dynamic K-factor (importance scaling)
        k_factor = g.get("importance", k)

        # Margin of victory adjustment
        margin = g.get("margin", 0)
        if margin_factor and margin:
            k_factor *= np.log(abs(margin) + 1)

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

    logger.info(f"Elo updated for {len(history)} games.")
    return elo, pd.DataFrame(history)


def championship_probabilities(elo, method="softmax", sims=1000, temperature=40.0, seed=None):
    """
    Convert Elo ratings to championship probabilities.

    Parameters:
        elo: dict team -> Elo rating
        method: "softmax" (approximation) or "simulation" (Monte Carlo playoffs)
        sims: number of simulations if method="simulation"
        temperature: scaling factor for softmax
        seed: random seed for reproducibility

    Returns:
        DataFrame with team, Elo rating, and championship probability
    """
    if not elo:
        logger.warning("No Elo ratings provided.")
        return pd.DataFrame()

    teams = list(elo.keys())
    ratings = np.array([elo[t] for t in teams], dtype=float)

    if method == "softmax":
        exps = np.exp((ratings - ratings.mean()) / temperature)
        probs = exps / exps.sum()
        df = pd.DataFrame({
            "team": teams,
            "elo": ratings,
            "championship_prob": probs
        }).sort_values("championship_prob", ascending=False)

    elif method == "simulation":
        if seed is not None:
            random.seed(seed)
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
        df = pd.DataFrame({
            "team": teams,
            "elo": ratings,
            "championship_prob": probs
        }).sort_values("championship_prob", ascending=False)

    else:
        raise ValueError("Invalid method. Choose 'softmax' or 'simulation'.")

    df["method"] = method
    df["generated_at"] = datetime.now().isoformat()
    logger.info(f"Championship probabilities computed using {method} method.")
    return df


def export_results(history: pd.DataFrame, champ_probs: pd.DataFrame, out_dir="results"):
    """
    Export Elo history and championship probabilities to CSV + metadata JSON.
    """
    os.makedirs(out_dir, exist_ok=True)

    if not history.empty:
        history_file = os.path.join(out_dir, "elo_history.csv")
        history.to_csv(history_file, index=False)
        logger.info(f"📊 Elo history exported to {history_file}")

    if not champ_probs.empty:
        champ_file = os.path.join(out_dir, "championship_probabilities.csv")
        champ_probs.to_csv(champ_file, index=False)
        logger.info(f"📊 Championship probabilities exported to {champ_file}")

    meta = {
        "generated_at": datetime.now().isoformat(),
        "history_rows": len(history),
        "championship_rows": len(champ_probs),
        "teams": champ_probs["team"].tolist() if not champ_probs.empty else []
    }
    meta_file = os.path.join(out_dir, "team_strength_meta.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"🧾 Metadata saved to {meta_file}")