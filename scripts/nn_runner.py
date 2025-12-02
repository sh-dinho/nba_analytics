# ============================================================
# File: scripts/nn_runner.py
# Neural Network Runner – loads ML & OU models, predicts outcomes,
# calculates EV & Kelly Criterion, prints colored output, returns structured results
# ============================================================

import copy
import numpy as np
import tensorflow as tf
from colorama import Fore, Style, init, deinit
from keras.models import load_model
from scripts.Utils import Expected_Value
from scripts.Utils import Kelly_Criterion as kc
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Initialize colorama for colored console output
init()

_model = None
_ou_model = None

def _load_models():
    """Lazy-load ML and OU models once."""
    global _model, _ou_model
    if _model is None:
        _model = load_model("Models/NN_Models/Trained-Model-ML-1699315388.285516")
        logger.info("✅ Moneyline model loaded")
    if _ou_model is None:
        _ou_model = load_model("Models/NN_Models/Trained-Model-OU-1699315414.2268295")
        logger.info("✅ Over/Under model loaded")

def nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion=True):
    """
    Run NN models for moneyline and over/under predictions.
    Returns structured results for downstream use (CSV, bankroll sim).
    """
    _load_models()

    # Batch predictions for efficiency
    ml_preds = _model.predict(np.array(data))
    frame_uo = copy.deepcopy(frame_ml)
    frame_uo["OU"] = np.asarray(todays_games_uo)
    ou_data = tf.keras.utils.normalize(frame_uo.values.astype(float), axis=1)
    ou_preds = _ou_model.predict(ou_data)

    results = []
    for i, (home_team, away_team) in enumerate(games):
        # Winner prediction
        winner_idx = int(np.argmax(ml_preds[i]))
        winner_conf = ml_preds[i][0][winner_idx] * 100

        # OU prediction
        ou_idx = int(np.argmax(ou_preds[i]))
        ou_conf = ou_preds[i][0][ou_idx] * 100

        # Expected values
        ev_home = ev_away = None
        if home_team_odds[i] and away_team_odds[i]:
            ev_home = Expected_Value.expected_value(ml_preds[i][0][1], int(home_team_odds[i]))
            ev_away = Expected_Value.expected_value(ml_preds[i][0][0], int(away_team_odds[i]))

        # Kelly fractions
        kelly_home = kc.calculate_kelly_criterion(home_team_odds[i], ml_preds[i][0][1]) if kelly_criterion else None
        kelly_away = kc.calculate_kelly_criterion(away_team_odds[i], ml_preds[i][0][0]) if kelly_criterion else None

        # Build structured result
        result = {
            "home_team": home_team,
            "away_team": away_team,
            "winner": home_team if winner_idx == 1 else away_team,
            "winner_confidence": round(winner_conf, 1),
            "ou_prediction": "UNDER" if ou_idx == 0 else "OVER",
            "ou_line": todays_games_uo[i],
            "ou_confidence": round(ou_conf, 1),
            "ev_home": round(ev_home, 3) if ev_home is not None else None,
            "ev_away": round(ev_away, 3) if ev_away is not None else None,
            "kelly_home": round(kelly_home, 3) if kelly_home is not None else None,
            "kelly_away": round(kelly_away, 3) if kelly_away is not None else None,
        }
        results.append(result)

        # Console output with colors
        winner_color = Fore.GREEN if winner_idx == 1 else Fore.RED
        ou_color = Fore.MAGENTA if ou_idx == 0 else Fore.BLUE
        print(
            f"{winner_color}{home_team}{Style.RESET_ALL} vs "
            f"{Fore.RED if winner_idx == 1 else Fore.GREEN}{away_team}{Style.RESET_ALL} "
            f"{Fore.CYAN}({round(winner_conf,1)}%){Style.RESET_ALL}: "
            f"{ou_color}{result['ou_prediction']} {Style.RESET_ALL}{todays_games_uo[i]} "
            f"{Fore.CYAN}({round(ou_conf,1)}%){Style.RESET_ALL}"
        )

        # EV + Kelly
        ev_home_color = Fore.GREEN if ev_home and ev_home > 0 else Fore.RED
        ev_away_color = Fore.GREEN if ev_away and ev_away > 0 else Fore.RED
        print(f"{home_team} EV: {ev_home_color}{ev_home}{Style.RESET_ALL}" +
              (f" Fraction of Bankroll: {result['kelly_home']}%" if kelly_home else ""))
        print(f"{away_team} EV: {ev_away_color}{ev_away}{Style.RESET_ALL}" +
              (f" Fraction of Bankroll: {result['kelly_away']}%" if kelly_away else ""))

    deinit()
    return results