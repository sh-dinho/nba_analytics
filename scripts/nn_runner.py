# ============================================================
# File: scripts/nn_runner.py
# Neural Network Runner – loads ML & OU models, predicts outcomes,
# calculates EV & Kelly Criterion, prints colored output, returns structured results
# ============================================================

from typing import List, Dict, Any, Optional
import numpy as np
from colorama import Fore, Style, init
from core.config import (
    MAX_KELLY_FRACTION,
    DEFAULT_BANKROLL,
    ML_MODEL_FILE_H5,    # use centralized ML model path
    OU_MODEL_FILE_H5,    # use centralized OU model path
    PRINT_ONLY_ACTIONABLE,
    EV_THRESHOLD,
    MIN_KELLY_STAKE,
)
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


def _validate_inputs(
    data: np.ndarray,
    todays_games_uo: List[float],
    games: List[Any],
    home_odds: List[float],
    away_odds: List[float],
    home_team: List[str],
    away_team: List[str],
) -> None:
    """Validate input arrays/lists for consistent lengths and basic sanity."""
    if data is None:
        raise PipelineError("Input 'data' is None.")
    if not isinstance(data, np.ndarray):
        raise PipelineError("Input 'data' must be a numpy.ndarray.")

    n = len(games)
    expected_lengths = {
        "todays_games_uo": len(todays_games_uo),
        "home_odds": len(home_odds),
        "away_odds": len(away_odds),
        "home_team": len(home_team),
        "away_team": len(away_team),
    }
    for name, l in expected_lengths.items():
        if l != n:
            raise PipelineError(f"Length mismatch: {name}={l}, games={n}")

    if data.shape[0] != n:
        raise PipelineError(f"Data batch size {data.shape[0]} does not match games length {n}")

    # Basic odds sanity checks
    for i, (ho, ao) in enumerate(zip(home_odds, away_odds)):
        if ho is None or ao is None:
            raise PipelineError(f"Odds cannot be None (index {i}).")
        if ho < 1.01 or ao < 1.01:
            logger.warning(f"⚠️ Unusual low decimal odds at index {i}: home={ho}, away={ao}")


def _softmax_if_needed(x: np.ndarray) -> np.ndarray:
    """Ensure binary outputs sum to 1. If not, apply softmax to logits."""
    s = float(np.sum(x))
    if np.min(x) >= 0.0 and np.max(x) <= 1.0 and 0.99 <= s <= 1.01:
        return x.astype(float)
    exps = np.exp(x - np.max(x))
    probs = exps / np.sum(exps)
    return probs.astype(float)


def _clip_prob(p: float, eps: float = 1e-6) -> float:
    """Avoid exact 0 or 1 probabilities to keep EV/Kelly stable."""
    return float(np.clip(p, eps, 1.0 - eps))


def _load_models() -> None:
    """Lazy-load ML and OU models once."""
    global _model, _ou_model
    if tf is None or load_model is None:
        raise PipelineError("TensorFlow not available to load NN models.")

    try:
        if _model is None:
            _model = load_model(ML_MODEL_FILE_H5)
            logger.info("✅ Moneyline model loaded")
        if _ou_model is None:
            if OU_MODEL_FILE_H5 is None:
                raise PipelineError("OU_MODEL_FILE_H5 not set in config.")
            _ou_model = load_model(OU_MODEL_FILE_H5)
            logger.info("✅ Over/Under model loaded")
    except Exception as e:
        logger.error(f"Failed to load NN models: {e}")
        raise PipelineError(f"Failed to load NN models: {e}")


def nn_runner(
    data: np.ndarray,
    todays_games_uo: List[float],
    games: List[Any],
    home_odds: List[float],
    away_odds: List[float],
    home_team: List[str],
    away_team: List[str],
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    Runs NN prediction and calculates betting metrics.

    Args:
        data: Feature data for prediction, shape (n_games, n_features).
        todays_games_uo: Over/Under lines per game.
        games: List of game identifiers.
        home_odds, away_odds: Decimal odds for home/away.
        home_team, away_team: Team names aligned to games.
        batch_size: Batch size for model prediction.

    Returns:
        A list of dictionaries with betting results and metrics.
    """
    if tf is None:
        logger.error("❌ NN Runner requires TensorFlow. Skipping execution.")
        return []

    _validate_inputs(data, todays_games_uo, games, home_odds, away_odds, home_team, away_team)
    _load_models()
    results: List[Dict[str, Any]] = []

    # Predictions
    try:
        predictions_ml = _model.predict(data, batch_size=batch_size)
    except Exception as e:
        logger.error(f"Moneyline prediction failed: {e}")
        raise PipelineError(f"Moneyline prediction failed: {e}")

    # OU may fail independently; continue with ML-only if needed
    predictions_ou: Optional[np.ndarray] = None
    try:
        predictions_ou = _ou_model.predict(data, batch_size=batch_size)
    except Exception as e:
        logger.warning(f"⚠️ OU prediction failed: {e}. Proceeding without OU.")

    # Validate output shapes: expect [n_games, 2]
    def _ensure_two_class_row(vec: np.ndarray, label: str, idx: int) -> np.ndarray:
        if vec.ndim != 1:
            raise PipelineError(f"{label} prediction row at index {idx} must be 1D, got shape {vec.shape}")
        if vec.size != 2:
            raise PipelineError(f"{label} prediction row at index {idx} must have 2 values, got {vec.size}")
        return vec

    for i, game in enumerate(games):
        ml_row = _ensure_two_class_row(predictions_ml[i], "ML", i)
        ml_probs = _softmax_if_needed(ml_row)

        # Convention: index 1 = home, index 0 = away
        home_prob = _clip_prob(float(ml_probs[1]))
        away_prob = _clip_prob(float(ml_probs[0]))

        # Winner prediction
        winner_idx = int(np.argmax(ml_probs))
        winner_conf = float(ml_probs[winner_idx] * 100.0)
        winner_prediction = home_team[i] if winner_idx == 1 else away_team[i]

        # OU prediction (optional)
        ou_prediction = None
        ou_conf = None
        if predictions_ou is not None:
            ou_row = _ensure_two_class_row(predictions_ou[i], "OU", i)
            ou_probs = _softmax_if_needed(ou_row)
            ou_idx = int(np.argmax(ou_probs))
            ou_conf = float(ou_probs[ou_idx] * 100.0)
            ou_prediction = "OVER" if ou_idx == 1 else "UNDER"

        # EV calculations (None-safe)
        ev_home = expected_value(home_prob, home_odds[i])
        ev_away = expected_value(away_prob, away_odds[i])

        # Kelly Criterion stake sizes with safety
        def _safe_kelly(odds: float, prob: float) -> float:
            try:
                stake = calculate_kelly_criterion(
                    odds,
                    prob,
                    bankroll=DEFAULT_BANKROLL,
                    max_fraction=MAX_KELLY_FRACTION,
                )
                return float(stake)
            except Exception:
                return 0.0

        kelly_home_stake = _safe_kelly(home_odds[i], home_prob)
        kelly_away_stake = _safe_kelly(away_odds[i], away_prob)

        # Actionable filter: only display if EV >= threshold and Kelly stake >= minimum
        actionable_home = (ev_home is not None and ev_home >= EV_THRESHOLD and kelly_home_stake >= MIN_KELLY_STAKE)
        actionable_away = (ev_away is not None and ev_away >= EV_THRESHOLD and kelly_away_stake >= MIN_KELLY_STAKE)
        is_actionable = actionable_home or actionable_away

        # Result row
        result = {
            "game_id": game,
            "home_team": home_team[i],
            "away_team": away_team[i],
            "home_prob": round(home_prob, 4),
            "away_prob": round(away_prob, 4),
            "winner_prediction": winner_prediction,
            "winner_conf_pct": round(winner_conf, 2),
            "ou_prediction": ou_prediction,
            "ou_conf_pct": round(ou_conf, 2) if ou_conf is not None else None,
            "ou_line": todays_games_uo[i],
            "home_odds_dec": round(float(home_odds[i]), 3),
            "away_odds_dec": round(float(away_odds[i]), 3),
            "ev_home": round(float(ev_home), 3) if ev_home is not None else None,
            "ev_away": round(float(ev_away), 3) if ev_away is not None else None,
            "kelly_home_stake": round(float(kelly_home_stake), 2),
            "kelly_away_stake": round(float(kelly_away_stake), 2),
            "actionable_home": actionable_home,
            "actionable_away": actionable_away,
            "is_actionable": is_actionable,
        }
        results.append(result)

        # Console output with colors (honor PRINT_ONLY_ACTIONABLE)
        if not PRINT_ONLY_ACTIONABLE or is_actionable:
            winner_color = Fore.GREEN if winner_idx == 1 else Fore.RED
            opp_color = Fore.RED if winner_idx == 1 else Fore.GREEN
            conf_str = f"{winner_conf:.1f}%"

            ou_text = ""
            if ou_prediction is not None and ou_conf is not None:
                ou_color = Fore.BLUE if ou_prediction == "OVER" else Fore.MAGENTA
                ou_text = f" {ou_color}{ou_prediction}{Style.RESET_ALL} {todays_games_uo[i]} {Fore.CYAN}({ou_conf:.1f}%){Style.RESET_ALL}"

            # Header line
            print(
                f"{winner_color}{home_team[i]}{Style.RESET_ALL} vs "
                f"{opp_color}{away_team[i]}{Style.RESET_ALL} "
                f"{Fore.CYAN}({conf_str}){Style.RESET_ALL}:{ou_text}"
            )

            # EV line
            def _fmt_ev(val: Optional[float]) -> str:
                if val is None:
                    return "N/A"
                return f"{val:.3f}"

            ev_home_color = Fore.GREEN if (ev_home is not None and ev_home > 0) else Fore.RED
            ev_away_color = Fore.GREEN if (ev_away is not None and ev_away > 0) else Fore.RED

            actionable_tag = ""
            if is_actionable:
                actionable_tag = f" {Fore.GREEN}[ACTIONABLE]{Style.RESET_ALL}"

            print(
                f"  > EV: {ev_home_color}H: {_fmt_ev(ev_home)}{Style.RESET_ALL} | "
                f"{ev_away_color}A: {_fmt_ev(ev_away)}{Style.RESET_ALL} "
                f"| Kelly Bet: H: ${kelly_home_stake:.2f} | A: ${kelly_away_stake:.2f}{actionable_tag}"
            )

    return results