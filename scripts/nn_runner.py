# ============================================================
# File: scripts/nn_runner.py
# Neural Network Runner – loads ML & OU models, predicts outcomes,
# calculates EV & Kelly Criterion, prints colored output,
# returns structured results, bankroll history, and summary metrics
# ============================================================

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from colorama import Fore, Style, init

from core.paths import ML_MODEL_FILE_H5, OU_MODEL_FILE_H5
from core.log_config import init_global_logger
from core.exceptions import PipelineError, DataError, FileError
from scripts.utils import expected_value, kelly_fraction
from core.config import (
    DEFAULT_BANKROLL,
    MAX_KELLY_FRACTION,
    PRINT_ONLY_ACTIONABLE,
    EV_THRESHOLD,
    MIN_KELLY_STAKE,
)

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    tf = None
    load_model = None

logger = init_global_logger()
init(autoreset=True)

_model = None
_ou_model = None


def _softmax_if_needed(x: np.ndarray) -> np.ndarray:
    """Ensure binary outputs sum to ~1; apply softmax if they don't."""
    s = float(np.sum(x))
    if np.min(x) >= 0.0 and np.max(x) <= 1.0 and 0.99 <= s <= 1.01:
        return x.astype(float)
    exps = np.exp(x - np.max(x))
    return (exps / np.sum(exps)).astype(float)


def _clip_prob(p: float, eps: float = 1e-6) -> float:
    """Avoid exact 0 or 1 for EV/Kelly stability."""
    return float(np.clip(p, eps, 1.0 - eps))


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
    if tf is None or load_model is None:
        raise PipelineError("TensorFlow not available to run NN models.")

    if data is None or not isinstance(data, np.ndarray):
        raise DataError("Input 'data' must be a numpy.ndarray and not None.")

    n = len(games)
    checks = {
        "todays_games_uo": len(todays_games_uo),
        "home_odds": len(home_odds),
        "away_odds": len(away_odds),
        "home_team": len(home_team),
        "away_team": len(away_team),
    }
    for name, l in checks.items():
        if l != n:
            raise DataError(f"Length mismatch: {name}={l}, games={n}")

    if data.shape[0] != n:
        raise DataError(f"Data rows ({data.shape[0]}) do not match games length ({n})")

    for i, (ho, ao) in enumerate(zip(home_odds, away_odds)):
        if ho is None or ao is None:
            raise DataError(f"Odds cannot be None (index {i})")
        if ho < 1.01 or ao < 1.01:
            logger.warning(f"⚠️ Unusually low decimal odds at index {i}: home={ho}, away={ao}")


def _ensure_two_class_row(vec: np.ndarray, label: str, idx: int) -> np.ndarray:
    if vec.ndim != 1 or vec.size != 2:
        raise PipelineError(f"{label} prediction row at index {idx} must be 1D of size 2, got shape {vec.shape}")
    return vec


def _load_models() -> None:
    """Lazy-load ML and OU models once, with existence checks."""
    global _model, _ou_model
    try:
        if _model is None:
            if not ML_MODEL_FILE_H5.exists():
                raise FileError("Moneyline NN model file not found", file_path=str(ML_MODEL_FILE_H5))
            _model = load_model(ML_MODEL_FILE_H5)
            logger.info(f"✅ Moneyline NN model loaded: {ML_MODEL_FILE_H5}")

        if _ou_model is None:
            if not OU_MODEL_FILE_H5.exists():
                raise FileError("OU NN model file not found", file_path=str(OU_MODEL_FILE_H5))
            _ou_model = load_model(OU_MODEL_FILE_H5)
            logger.info(f"✅ Over/Under NN model loaded: {OU_MODEL_FILE_H5}")
    except Exception as e:
        logger.error(f"❌ Failed to load NN models: {e}")
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
    use_kelly: bool = True,
    bankroll: float = DEFAULT_BANKROLL,
    max_fraction: float = MAX_KELLY_FRACTION,
) -> Tuple[List[Dict[str, Any]], List[float], Dict[str, float]]:
    """
    Run NN models for moneyline and optional OU predictions.
    Simulates bankroll trajectory using EV and Kelly bet sizes.

    Returns:
        - results: list of dicts with predictions and betting metrics
        - history: bankroll after each game (simulated)
        - metrics: summary dict (final bankroll, avg EV, win rate, total bets, actionable count)
    """
    _validate_inputs(data, todays_games_uo, games, home_odds, away_odds, home_team, away_team)
    _load_models()

    # Predictions
    try:
        predictions_ml = _model.predict(data, batch_size=batch_size)
    except Exception as e:
        raise PipelineError(f"Moneyline prediction failed: {e}")

    predictions_ou: Optional[np.ndarray] = None
    try:
        predictions_ou = _ou_model.predict(data, batch_size=batch_size)
    except Exception as e:
        logger.warning(f"⚠️ OU prediction failed: {e}. Proceeding without OU.")

    results: List[Dict[str, Any]] = []
    history: List[float] = []
    current_bankroll = float(bankroll)

    actionable_count = 0
    bet_count = 0
    ev_values: List[float] = []
    win_count = 0  # simulated win count for ML bets

    for i, game in enumerate(games):
        ml_row = _ensure_two_class_row(predictions_ml[i], "ML", i)
        ml_probs = _softmax_if_needed(ml_row)
        home_prob = _clip_prob(float(ml_probs[1]))
        away_prob = _clip_prob(float(ml_probs[0]))

        # Winner prediction
        winner_idx = int(np.argmax(ml_probs))
        winner_conf = float(ml_probs[winner_idx] * 100.0)
        winner_prediction = home_team[i] if winner_idx == 1 else away_team[i]

        # OU prediction (optional)
        ou_prediction, ou_conf = None, None
        if predictions_ou is not None:
            ou_row = _ensure_two_class_row(predictions_ou[i], "OU", i)
            ou_probs = _softmax_if_needed(ou_row)
            ou_idx = int(np.argmax(ou_probs))
            ou_conf = float(ou_probs[ou_idx] * 100.0)
            ou_prediction = "OVER" if ou_idx == 1 else "UNDER"

        # EV
        ev_home = expected_value(home_prob, float(home_odds[i]))
        ev_away = expected_value(away_prob, float(away_odds[i]))
        if ev_home is not None:
            ev_values.append(ev_home)
        if ev_away is not None:
            ev_values.append(ev_away)

        # Kelly fractions and bet sizes
        k_home = kelly_fraction(home_prob, float(home_odds[i])) if use_kelly else None
        k_away = kelly_fraction(away_prob, float(away_odds[i])) if use_kelly else None

        frac_home = min(k_home if (k_home is not None and k_home > 0) else max_fraction, max_fraction) if use_kelly else max_fraction
        frac_away = min(k_away if (k_away is not None and k_away > 0) else max_fraction, max_fraction) if use_kelly else max_fraction
        bet_size_home = current_bankroll * frac_home
        bet_size_away = current_bankroll * frac_away

        actionable_home = (ev_home is not None and ev_home >= EV_THRESHOLD and bet_size_home >= MIN_KELLY_STAKE)
        actionable_away = (ev_away is not None and ev_away >= EV_THRESHOLD and bet_size_away >= MIN_KELLY_STAKE)
        is_actionable = actionable_home or actionable_away
        actionable_count += int(is_actionable)

        # Simulate ML outcomes (Monte Carlo using predicted probs)
        rng = np.random.default_rng()
        outcome_home = "WIN" if rng.random() < home_prob else "LOSS"
        profit_home = bet_size_home * (float(home_odds[i]) - 1.0) if outcome_home == "WIN" else -bet_size_home

        outcome_away = "WIN" if rng.random() < away_prob else "LOSS"
        profit_away = bet_size_away * (float(away_odds[i]) - 1.0) if outcome_away == "WIN" else -bet_size_away

        bet_count += 2
        win_count += int(outcome_home == "WIN") + int(outcome_away == "WIN")

        current_bankroll += (profit_home + profit_away)
        history.append(round(current_bankroll, 2))

        result = {
            "game_id": game,
            "home_team": home_team[i],
            "away_team": away_team[i],
            "winner_prediction": winner_prediction,
            "winner_conf_pct": round(winner_conf, 1),
            "ou_prediction": ou_prediction,
            "ou_line": todays_games_uo[i],
            "ou_conf_pct": round(ou_conf, 1) if ou_conf is not None else None,
            "home_prob": round(home_prob, 4),
            "away_prob": round(away_prob, 4),
            "ev_home": round(ev_home, 3) if ev_home is not None else None,
            "ev_away": round(ev_away, 3) if ev_away is not None else None,
            "kelly_home": round(k_home, 3) if k_home is not None else None,
            "kelly_away": round(k_away, 3) if k_away is not None else None,
            "bet_size_home": round(bet_size_home, 2),
            "bet_size_away": round(bet_size_away, 2),
            "outcome_home": outcome_home,
            "outcome_away": outcome_away,
            "profit_home": round(profit_home, 2),
            "profit_away": round(profit_away, 2),
            "bankroll_after": round(current_bankroll, 2),
            "actionable_home": actionable_home,
            "actionable_away": actionable_away,
            "is_actionable": is_actionable,
        }
        results.append(result)

        # Console output (honor PRINT_ONLY_ACTIONABLE)
        if not PRINT_ONLY_ACTIONABLE or is_actionable:
            winner_color = Fore.GREEN if winner_idx == 1 else Fore.RED
            opp_color = Fore.RED if winner_idx == 1 else Fore.GREEN
            ou_text = ""
            if ou_prediction is not None and ou_conf is not None:
                ou_color = Fore.BLUE if ou_prediction == "OVER" else Fore.MAGENTA
                ou_text = f" | {ou_color}{ou_prediction}{Style.RESET_ALL} {todays_games_uo[i]} ({ou_conf:.1f}%)"

            print(
                f"[{'ACTIONABLE' if is_actionable else 'INFO'}] "
                f"{winner_color}{home_team[i]}{Style.RESET_ALL} vs {opp_color}{away_team[i]}{Style.RESET_ALL} "
                f"| Winner {winner_prediction} ({winner_conf:.1f}%)"
                f"{ou_text} | EV H:{result['ev_home']} A:{result['ev_away']} "
                f"| Kelly H:{result['kelly_home']} A:{result['kelly_away']} "
                f"| Bet H:{result['bet_size_home']} A:{result['bet_size_away']}"
            )

    # Summary metrics
    avg_ev = float(np.mean(ev_values)) if ev_values else 0.0
    win_rate = (win_count / bet_count) if bet_count > 0 else 0.0
    metrics = {
        "final_bankroll": round(current_bankroll, 2),
        "avg_ev": round(avg_ev, 4),
        "win_rate": round(win_rate, 4),
        "total_bets": int(bet_count),
        "actionable_count": int(actionable_count),
    }

    logger.info(
        f"📊 Summary — Final bankroll: {metrics['final_bankroll']}, "
        f"Avg EV: {metrics['avg_ev']}, Win rate: {metrics['win_rate']}, "
        f"Bets: {metrics['total_bets']}, Actionable: {metrics['actionable_count']}"
    )

    return results, history, metrics
