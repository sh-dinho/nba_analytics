# ============================================================
# File: scripts/nn_runner.py
# Neural Network Runner – loads ML & OU models, predicts outcomes,
# calculates EV & Kelly Criterion, prints colored output, returns structured results
# ============================================================

import copy
import numpy as np
import logging
from colorama import Fore, Style, init, deinit

# --- FIX: New Unified Imports ---
from betting.utils import expected_value, calculate_kelly_fraction
from core.config import MAX_KELLY_FRACTION # Use centralized Max Kelly Fraction

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    tf = None
    load_model = None
    print("WARNING: TensorFlow not installed. NN model functionality will be disabled.")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Initialize colorama for colored console output
init()

_model = None
_ou_model = None
# Default bankroll for calculating Kelly stake size in runner output
DEFAULT_BANKROLL_FOR_DISPLAY = 1000.0 

def _load_models():
    """Lazy-load ML and OU models once. Paths remain hardcoded for example in NN runner."""
    global _model, _ou_model
    if tf is None:
        raise ImportError("TensorFlow not available to load NN models.")
        
    if _model is None:
        # Assuming these paths are correct for the original NN setup
        _model = load_model("Models/NN_Models/Trained-Model-ML-1699315388.285516") 
        logger.info("✅ Moneyline model loaded")
    if _ou_model is None:
        _ou_model = load_model("Models/NN_Models/Trained-Model-OU-1699315414.2268295")
        logger.info("✅ Over/Under model loaded")

def nn_runner(data, todays_games_uo, frame_ml, games, home_odds, away_odds, home_uo, away_uo, home_team, away_team):
    """
    Runs NN prediction and calculates betting metrics.
    
    Args:
        data: Feature data for prediction.
        todays_games_uo: Over/Under lines.
        frame_ml: Data frame containing home/away team info.
        games: List of game IDs.
        home_odds, away_odds: List of decimal odds for home/away.
        home_uo, away_uo: List of over/under probabilities (if available).
        home_team, away_team: List of home and away team names.

    Returns:
        A list of dictionaries with betting results and metrics.
    """
    if tf is None:
        logger.error("NN Runner requires TensorFlow. Skipping execution.")
        return []
        
    _load_models()
    results = []

    # Make predictions
    predictions_ml = _model.predict(data)
    predictions_ou = _ou_model.predict(data)
    
    for i, game in enumerate(games):
        # Extract probabilities
        home_prob = predictions_ml[i][1]
        away_prob = predictions_ml[i][0]
        
        winner_idx = np.argmax(predictions_ml[i])
        winner_conf = predictions_ml[i][winner_idx] * 100
        
        ou_idx = np.argmax(predictions_ou[i])
        ou_conf = predictions_ou[i][ou_idx] * 100
        ou_prediction = "OVER" if ou_idx == 1 else "UNDER"

        # Extract decimal odds
        home_odds_dec = home_odds[i]
        away_odds_dec = away_odds[i]
        
        # ------------------------------------------------
        # CRITICAL FIX: Use the new unified utility functions
        # ------------------------------------------------
        
        # Calculate EV (Expected Value)
        # Note: betting/utils.py's expected_value uses a unit stake (1.0)
        ev_home = expected_value(home_prob, home_odds_dec)
        ev_away = expected_value(away_prob, away_odds_dec)
        
        # Calculate Kelly Criterion Fraction
        kelly_home_fraction = calculate_kelly_fraction(home_prob, home_odds_dec, MAX_KELLY_FRACTION)
        kelly_away_fraction = calculate_kelly_fraction(away_prob, away_odds_dec, MAX_KELLY_FRACTION)
        
        # Convert Kelly Fraction to a Stake size for display
        kelly_home_stake = kelly_home_fraction * DEFAULT_BANKROLL_FOR_DISPLAY
        kelly_away_stake = kelly_away_fraction * DEFAULT_BANKROLL_FOR_DISPLAY

        result = {
            "game_id": game,
            "home_team": home_team[i],
            "away_team": away_team[i],
            "home_prob": round(home_prob, 4),
            "away_prob": round(away_prob, 4),
            "winner_prediction": frame_ml.iloc[i].iloc[winner_idx + 1], # Assuming team names are at index 1 and 2
            "ou_prediction": ou_prediction,
            "ou_line": todays_games_uo[i],
            "home_odds_dec": round(home_odds_dec, 3),
            "away_odds_dec": round(away_odds_dec, 3),
            # FIX: EV and Kelly results updated
            "ev_home": round(ev_home, 3) if ev_home is not None else None,
            "ev_away": round(ev_away, 3) if ev_away is not None else None,
            "kelly_home_fraction": round(kelly_home_fraction, 5),
            "kelly_away_fraction": round(kelly_away_fraction, 5),
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
            f"{Fore.CYAN}({round(winner_conf,1)}%){Style.RESET_ALL}: "
            f"{ou_color}{result['ou_prediction']} {Style.RESET_ALL}{todays_games_uo[i]} "
            f"{Fore.CYAN}({round(ou_conf,1)}%){Style.RESET_ALL}"
        )

        # EV + Kelly output
        ev_home_color = Fore.GREEN if ev_home and ev_home > 0 else Fore.RED
        ev_away_color = Fore.GREEN if ev_away and ev_away > 0 else Fore.RED
        
        print(
            f"  > EV: {ev_home_color}H: {ev_home:.3f}{Style.RESET_ALL} | "
            f"{ev_away_color}A: {ev_away:.3f}{Style.RESET_ALL} "
            f"| Kelly Bet: H: ${kelly_home_stake:.2f} | A: ${kelly_away_stake:.2f}"
        )

    deinit()
    return results