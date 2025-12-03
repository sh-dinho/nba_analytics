# ============================================================
# File: scripts/nn_runner.py
# Neural Network Runner – loads ML & OU models, predicts outcomes,
# calculates EV & Kelly Criterion, prints colored output, returns structured results
# ============================================================

import numpy as np
from colorama import Fore, Style, init
from core.config import MAX_KELLY_FRACTION, DEFAULT_BANKROLL, MODEL_FILE_H5
from core.log_config import setup_logger
from core.exceptions import PipelineError
from betting.utils import expected_value, calculate_kelly_criterion

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    tf = None
    load_model = None

logger = setup_logger("nn_runner")

# Initialize colorama once globally
init(autoreset=True)

_model = None
_ou_model = None


def _load_models():
    """Lazy-load ML and OU models once."""
    global _model, _ou_model
    if tf is None:
        raise PipelineError("TensorFlow not available to load NN models.")

    try:
        if _model is None:
            _model = load_model(MODEL_FILE_H5)  # centralized path
            logger.info("✅ Moneyline model loaded")
        if _ou_model is None:
            # Example: you may want to add OU_MODEL_FILE_H5 in config
            _ou_model = load_model("Models/NN_Models/Trained-Model-OU.h5")
            logger.info("✅ Over/Under model loaded")
    except Exception as e:
        logger.error(f"Failed to load NN models: {e}")
        raise PipelineError(f"Failed to load NN models: {e}")


def nn_runner(data, todays_games_uo, games,
              home_odds, away_odds, home_team, away_team):
    """
    Runs NN prediction and calculates betting metrics.

    Args:
        data: Feature data for prediction.
        todays_games_uo: Over/Under lines.
        games: List of game IDs.
        home_odds, away_odds: List of decimal odds for home/away.
        home_team, away_team: List of home and away team names.

    Returns:
        A list of dictionaries with betting results and metrics.
    """
    if tf is None:
        logger.error("❌ NN Runner requires TensorFlow. Skipping execution.")
        return []

    _load_models()
    results = []

    try:
        predictions_ml = _model.predict(data, batch_size=32)
        predictions_ou = _ou_model.predict(data, batch_size=32)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise PipelineError(f"Prediction failed: {e}")

    for i, game in enumerate(games):
        home_prob, away_prob = predictions_ml[i][1], predictions_ml[i][0]

        # Winner prediction
        winner_idx = np.argmax(predictions_ml[i])
        winner_conf = predictions_ml[i][winner_idx] * 100
        winner_prediction = home_team[i] if winner_idx == 1 else away_team[i]

        # Over/Under prediction
        ou_idx = np.argmax(predictions_ou[i])
        ou_conf = predictions_ou[i][ou_idx] * 100
        ou_prediction = "OVER" if ou_idx == 1 else "UNDER"

        # EV calculations
        ev_home = expected_value(home_prob, home_odds[i])
        ev_away = expected_value(away_prob, away_odds[i])

        # Kelly Criterion stake sizes
        kelly_home_stake = calculate_kelly_criterion(home_odds[i], home_prob,
                                                     bankroll=DEFAULT_BANKROLL,
                                                     max_fraction=MAX_KELLY_FRACTION)
        kelly_away_stake = calculate_kelly_criterion(away_odds[i], away_prob,
                                                     bankroll=DEFAULT_BANKROLL,
                                                     max_fraction=MAX_KELLY_FRACTION)

        result = {
            "game_id": game,
            "home_team": home_team[i],
            "away_team": away_team[i],
            "home_prob": round(home_prob, 4),
            "away_prob": round(away_prob, 4),
            "winner_prediction": winner_prediction,
            "ou_prediction": ou_prediction,
            "ou_line": todays_games_uo[i],
            "home_odds_dec": round(home_odds[i], 3),
            "away_odds_dec": round(away_odds[i], 3),
            "ev_home": round(ev_home, 3) if ev_home is not None else None,
            "ev_away": round(ev_away, 3) if ev_away is not None else None,
            "kelly_home_stake": round(kelly_home_stake, 2),
            "kelly_away_stake": round(kelly_away_stake, 2),
        }
        results.append(result)

        # Console output with colors
        winner_color = Fore.GREEN if winner_idx == 1 else Fore.RED
        ou_color = Fore.MAGENTA if ou_idx == 0 else Fore.BLUE
        print(
            f"{winner_color}{home_team[i]}{Style.RESET_ALL} vs "
            f"{Fore.RED if winner_idx == 1 else Fore.GREEN}{away_team[i]}{Style.RESET_ALL} "
            f"{Fore.CYAN}({winner_conf:.1f}%){Style.RESET_ALL}: "
            f"{ou_color}{ou_prediction}{Style.RESET_ALL} {todays_games_uo[i]} "
            f"{Fore.CYAN}({ou_conf:.1f}%){Style.RESET_ALL}"
        )

        ev_home_color = Fore.GREEN if ev_home and ev_home > 0 else Fore.RED
        ev_away_color = Fore.GREEN if ev_away and ev_away > 0 else Fore.RED
        print(
            f"  > EV: {ev_home_color}H: {ev_home:.3f}{Style.RESET_ALL} | "
            f"{ev_away_color}A: {ev_away:.3f}{Style.RESET_ALL} "
            f"| Kelly Bet: H: ${kelly_home_stake:.2f} | A: ${kelly_away_stake:.2f}"
        )

    return results