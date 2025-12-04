# ============================================================
# File: scripts/xgb_runner.py
# XGBoost Runner – loads ML & OU models, predicts outcomes,
# calculates EV & Kelly Criterion, simulates bankroll trajectory,
# prints colored output, returns structured results
# ============================================================

from typing import List, Tuple, Any, Dict, Optional
import copy
import numpy as np
import xgboost as xgb
import random
from colorama import Fore, Style, init

from scripts.utils import expected_value, kelly_fraction
from core.log_config import init_global_logger
from core.exceptions import PipelineError, DataError, FileError
from core.paths import (
    ARCHIVE_DIR,
)
from core.config import (
    XGB_ML_MODEL_FILE,
    XGB_OU_MODEL_FILE,
    DEFAULT_BANKROLL,
    MAX_KELLY_FRACTION,
    PRINT_ONLY_ACTIONABLE,
    EV_THRESHOLD,
    MIN_KELLY_STAKE,
)

logger = init_global_logger()
init(autoreset=True)

# Load models once with defensive checks
try:
    if not XGB_ML_MODEL_FILE.exists():
        raise FileError(f"Moneyline model not found: {XGB_ML_MODEL_FILE}", file_path=str(XGB_ML_MODEL_FILE))
    if not XGB_OU_MODEL_FILE.exists():
        raise FileError(f"Over/Under model not found: {XGB_OU_MODEL_FILE}", file_path=str(XGB_OU_MODEL_FILE))

    xgb_ml = xgb.Booster()
    xgb_ml.load_model(str(XGB_ML_MODEL_FILE))
    logger.info(f"✅ Moneyline XGBoost model loaded: {XGB_ML_MODEL_FILE}")

    xgb_ou = xgb.Booster()
    xgb_ou.load_model(str(XGB_OU_MODEL_FILE))
    logger.info(f"✅ Over/Under XGBoost model loaded: {XGB_OU_MODEL_FILE}")
except Exception as e:
    logger.error(f"❌ Failed to load XGBoost models: {e}")
    raise PipelineError(f"Model loading failed: {e}")


def _softmax_row(row: np.ndarray) -> np.ndarray:
    """Convert logits or arbitrary scores to probabilities (2-class)."""
    if row.ndim != 1 or row.size != 2:
        raise PipelineError(f"Expected a 1D vector of size 2, got shape {row.shape}")
    s = float(np.sum(row))
    # Already probabilities?
    if np.min(row) >= 0.0 and np.max(row) <= 1.0 and 0.99 <= s <= 1.01:
        return row.astype(float)
    exps = np.exp(row - np.max(row))
    probs = exps / np.sum(exps)
    return probs.astype(float)


def _clip_prob(p: float, eps: float = 1e-6) -> float:
    """Avoid exact 0 or 1 for EV/Kelly stability."""
    return float(np.clip(p, eps, 1.0 - eps))


def _validate_inputs(
    data: np.ndarray,
    todays_games_uo: List[float],
    frame_ml,
    games: List[Tuple[str, str]],
    home_team_odds: List[float],
    away_team_odds: List[float],
):
    n_games = len(games)
    if n_games == 0:
        raise DataError("No games provided to xgb_runner.")

    if data is None:
        raise DataError("Feature matrix 'data' is None.")
    if np.asarray(data).shape[0] != n_games:
        raise DataError(f"Feature rows ({np.asarray(data).shape[0]}) != number of games ({n_games}).")

    if len(todays_games_uo) != n_games:
        raise DataError(f"OU lines length ({len(todays_games_uo)}) != number of games ({n_games}).")

    if len(home_team_odds) != n_games or len(away_team_odds) != n_games:
        raise DataError("Home/Away odds lengths must match number of games.")

    if getattr(frame_ml, "shape", None) is None or frame_ml.shape[0] != n_games:
        raise DataError("frame_ml must be a DataFrame aligned to games (same number of rows).")


def xgb_runner(
    data: np.ndarray,
    todays_games_uo: List[float],
    frame_ml,
    games: List[Tuple[str, str]],
    home_team_odds: List[float],
    away_team_odds: List[float],
    use_kelly: bool = True,
    bankroll: float = DEFAULT_BANKROLL,
    max_fraction: float = MAX_KELLY_FRACTION,
    seed: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[float], Dict[str, float]]:
    """
    Run XGBoost models for moneyline and over/under predictions.
    Simulates bankroll trajectory using EV and Kelly bet sizes.

    Args:
        data: Feature matrix for ML predictions (n_games, n_features), aligned to 'games'.
        todays_games_uo: List of OU lines (length n_games).
        frame_ml: DataFrame with team info aligned to games rows.
        games: List of (home_team, away_team) tuples.
        home_team_odds, away_team_odds: Lists of decimal odds aligned to games.
        use_kelly: Whether to apply Kelly criterion.
        bankroll: Starting bankroll.
        max_fraction: Max fraction of bankroll to risk per bet.
        seed: Optional random seed for reproducibility.

    Returns:
        results: List of dicts with predictions and betting metrics.
        history: Bankroll trajectory after each game.
        metrics: Summary dict (final bankroll, avg EV, win rate, total bets).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    _validate_inputs(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds)

    try:
        ml_preds_raw = xgb_ml.predict(xgb.DMatrix(np.asarray(data, dtype=float)))
        ml_preds = np.asarray(ml_preds_raw).reshape(len(games), -1)

        frame_uo = copy.deepcopy(frame_ml)
        frame_uo = frame_uo.copy()
        frame_uo["OU"] = np.asarray(todays_games_uo, dtype=float)
        ou_data = frame_uo.values.astype(float)

        ou_preds_raw = xgb_ou.predict(xgb.DMatrix(ou_data))
        ou_preds = np.asarray(ou_preds_raw).reshape(len(games), -1)
    except Exception as e:
        raise PipelineError(f"Prediction failed: {e}")

    results: List[Dict[str, Any]] = []
    history: List[float] = []
    current_bankroll = float(bankroll)

    actionable_count = 0
    bet_count = 0
    ev_values: List[float] = []
    win_count = 0  # simulated win count for ML bets

    for i, (home_team, away_team) in enumerate(games):
        # Moneyline probs (two-class: [away_prob, home_prob] or similar)
        ml_probs = _softmax_row(ml_preds[i])
        home_prob = _clip_prob(float(ml_probs[1]))
        away_prob = _clip_prob(float(ml_probs[0]))

        # Winner prediction
        winner_idx = int(np.argmax(ml_probs))
        winner_conf = float(ml_probs[winner_idx] * 100.0)
        predicted_winner = home_team if winner_idx == 1 else away_team

        # OU probs
        ou_probs = _softmax_row(ou_preds[i])
        ou_idx = int(np.argmax(ou_probs))
        ou_conf = float(ou_probs[ou_idx] * 100.0)
        ou_prediction = "OVER" if ou_idx == 1 else "UNDER"

        # EV
        ev_home = expected_value(home_prob, float(home_team_odds[i]))
        ev_away = expected_value(away_prob, float(away_team_odds[i]))

        # Record EV sample for summary
        if ev_home is not None:
            ev_values.append(ev_home)
        if ev_away is not None:
            ev_values.append(ev_away)

        # Kelly fractions
        k_home = kelly_fraction(home_prob, float(home_team_odds[i])) if use_kelly else None
        k_away = kelly_fraction(away_prob, float(away_team_odds[i])) if use_kelly else None

        # Bet sizes
        frac_home = min(k_home if (k_home is not None and k_home > 0) else max_fraction, max_fraction) if use_kelly else max_fraction
        frac_away = min(k_away if (k_away is not None and k_away > 0) else max_fraction, max_fraction) if use_kelly else max_fraction
        bet_size_home = current_bankroll * frac_home
        bet_size_away = current_bankroll * frac_away

        # Actionable filter
        actionable_home = (ev_home is not None and ev_home >= EV_THRESHOLD and bet_size_home >= MIN_KELLY_STAKE)
        actionable_away = (ev_away is not None and ev_away >= EV_THRESHOLD and bet_size_away >= MIN_KELLY_STAKE)
        is_actionable = actionable_home or actionable_away
        actionable_count += int(is_actionable)

        # Simulate outcomes (Monte Carlo based on predicted probs)
        outcome_home = "WIN" if random.random() < home_prob else "LOSS"
        profit_home = bet_size_home * (float(home_team_odds[i]) - 1.0) if outcome_home == "WIN" else -bet_size_home

        outcome_away = "WIN" if random.random() < away_prob else "LOSS"
        profit_away = bet_size_away * (float(away_team_odds[i]) - 1.0) if outcome_away == "WIN" else -bet_size_away

        bet_count += 2  # tracking bets placed (home and away legs)
        win_count += int(outcome_home == "WIN") + int(outcome_away == "WIN")

        current_bankroll += (profit_home + profit_away)
        history.append(round(current_bankroll, 2))

        result = {
            "home_team": home_team,
            "away_team": away_team,
            "winner": predicted_winner,
            "winner_confidence_pct": round(winner_conf, 1),
            "ou_prediction": ou_prediction,
            "ou_line": todays_games_uo[i],
            "ou_confidence_pct": round(ou_conf, 1),
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

        # Console output (colored)
        if not PRINT_ONLY_ACTIONABLE or is_actionable:
            winner_color = Fore.GREEN if winner_idx == 1 else Fore.RED
            opp_color = Fore.RED if winner_idx == 1 else Fore.GREEN
            prefix = f"{home_team} vs {away_team}"
            winner_str = f"Winner: {predicted_winner} ({winner_conf:.1f}%)"
            ou_str = f"OU: {ou_prediction} {todays_games_uo[i]} ({ou_conf:.1f}%)"
            ev_str = f"EV H:{result['ev_home']} A:{result['ev_away']}"
            kelly_str = f"K H:{result['kelly_home']} A:{result['kelly_away']}"
            bet_str = f"Bet H:{result['bet_size_home']} A:{result['bet_size_away']}"
            act_str = "ACTIONABLE" if is_actionable else "INFO"
            color = Fore.CYAN if is_actionable else Fore.WHITE

            print(color + f"[{act_str}] {prefix} | "
                  + winner_color + winner_str + Style.RESET_ALL + " | "
                  + opp_color + ou_str + Style.RESET_ALL + " | "
                  + Fore.YELLOW + ev_str + Style.RESET_ALL + " | "
                  + Fore.MAGENTA + kelly_str + Style.RESET_ALL + " | "
                  + Fore.BLUE + bet_str + Style.RESET_ALL)

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

    logger.info(f"📊 Summary — Final bankroll: {metrics['final_bankroll']}, "
                f"Avg EV: {metrics['avg_ev']}, Win rate: {metrics['win_rate']}, "
                f"Bets: {metrics['total_bets']}, Actionable: {metrics['actionable_count']}")

    return results, history, metrics
