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
from core.log_config import setup_logger
from core.exceptions import PipelineError
from core.config import (
    XGB_ML_MODEL_FILE,
    XGB_OU_MODEL_FILE,
    DEFAULT_BANKROLL,
    MAX_KELLY_FRACTION,
    PRINT_ONLY_ACTIONABLE,
    EV_THRESHOLD,
    MIN_KELLY_STAKE,
)

logger = setup_logger("xgb_runner")
init(autoreset=True)

# Load models once
try:
    xgb_ml = xgb.Booster()
    xgb_ml.load_model(str(XGB_ML_MODEL_FILE))
    logger.info("✅ Moneyline XGBoost model loaded")

    xgb_ou = xgb.Booster()
    xgb_ou.load_model(str(XGB_OU_MODEL_FILE))
    logger.info("✅ Over/Under XGBoost model loaded")
except Exception as e:
    logger.error(f"❌ Failed to load XGBoost models: {e}")
    raise PipelineError(f"Model loading failed: {e}")


def _softmax_row(row: np.ndarray) -> np.ndarray:
    """Convert logits or arbitrary scores to probabilities (2-class)."""
    if row.ndim != 1 or row.size != 2:
        raise PipelineError(f"Expected a 1D vector of size 2, got shape {row.shape}")
    s = float(np.sum(row))
    if np.min(row) >= 0.0 and np.max(row) <= 1.0 and 0.99 <= s <= 1.01:
        return row.astype(float)
    exps = np.exp(row - np.max(row))
    probs = exps / np.sum(exps)
    return probs.astype(float)


def _clip_prob(p: float, eps: float = 1e-6) -> float:
    """Avoid exact 0 or 1 for EV/Kelly stability."""
    return float(np.clip(p, eps, 1.0 - eps))


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
        data: Feature matrix for ML predictions (n_games, n_features).
        todays_games_uo: List of OU lines (length n_games).
        frame_ml: DataFrame with team info aligned to games.
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

    try:
        ml_preds_raw = xgb_ml.predict(xgb.DMatrix(np.asarray(data, dtype=float)))
        ml_preds = np.asarray(ml_preds_raw).reshape(len(games), -1)

        frame_uo = copy.deepcopy(frame_ml)
        frame_uo["OU"] = np.asarray(todays_games_uo)
        ou_data = frame_uo.values.astype(float)

        ou_preds_raw = xgb_ou.predict(xgb.DMatrix(ou_data))
        ou_preds = np.asarray(ou_preds_raw).reshape(len(games), -1)
    except Exception as e:
        raise PipelineError(f"Prediction failed: {e}")

    results: List[Dict[str, Any]] = []
    history: List[float] = []
    current_bankroll = float(bankroll)

    for i, (home_team, away_team) in enumerate(games):
        # Moneyline probs
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

        # Simulate outcomes
        outcome_home = "WIN" if random.random() < home_prob else "LOSS"
        profit_home = bet_size_home * (float(home_team_odds[i]) - 1.0) if outcome_home == "WIN" else -bet_size_home

        outcome_away = "WIN" if random.random() < away_prob else "LOSS"
        profit_away = bet_size_away * (float(away_team_odds[i]) - 1.0) if outcome_away == "WIN" else -bet_size_away

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

        # Console output
        if not PRINT_ONLY_ACTIONABLE or is_actionable:
            winner_color = Fore.GREEN if winner_idx == 1 else Fore.RED
            opp_color = Fore.RED if winner_idx == 1 else Fore.GREEN