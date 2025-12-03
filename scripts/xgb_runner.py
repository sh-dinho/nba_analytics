# ============================================================
# File: scripts/xgb_runner.py
# XGBoost Runner – loads ML & OU models, predicts outcomes,
# calculates EV & Kelly Criterion, simulates bankroll trajectory,
# prints colored output, returns structured results
# ============================================================

import copy
import numpy as np
import xgboost as xgb
import random
from colorama import Fore, Style
from scripts.utils import expected_value, kelly_fraction
from core.log_config import setup_logger
from core.exceptions import PipelineError
from core.config import XGB_ML_MODEL_FILE, XGB_OU_MODEL_FILE

logger = setup_logger("xgb_runner")

# Load models once
try:
    xgb_ml = xgb.Booster()
    xgb_ml.load_model(XGB_ML_MODEL_FILE)
    logger.info("✅ Moneyline XGBoost model loaded")

    xgb_uo = xgb.Booster()
    xgb_uo.load_model(XGB_OU_MODEL_FILE)
    logger.info("✅ Over/Under XGBoost model loaded")
except Exception as e:
    logger.error(f"❌ Failed to load XGBoost models: {e}")
    raise PipelineError(f"Model loading failed: {e}")


def xgb_runner(data, todays_games_uo, frame_ml, games,
               home_team_odds, away_team_odds,
               use_kelly=True, bankroll=1000.0, max_fraction=0.05, seed=None):
    """
    Run XGBoost models for moneyline and over/under predictions.
    Simulates bankroll trajectory using EV and Kelly bet sizes.

    Args:
        data: Feature matrix for ML predictions.
        todays_games_uo: List of OU lines.
        frame_ml: DataFrame with team info.
        games: List of (home_team, away_team) tuples.
        home_team_odds, away_team_odds: Lists of decimal odds.
        use_kelly: Whether to apply Kelly criterion.
        bankroll: Starting bankroll.
        max_fraction: Max fraction of bankroll to risk per bet.
        seed: Optional random seed for reproducibility.

    Returns:
        results: List of dicts with predictions and betting metrics.
        history: Bankroll trajectory after each bet.
        metrics: Summary dict (final bankroll, avg EV, win rate).
    """
    if seed is not None:
        random.seed(seed)

    try:
        ml_preds = np.array(xgb_ml.predict(xgb.DMatrix(np.array(data)))).reshape(len(games), -1)
        frame_uo = copy.deepcopy(frame_ml)
        frame_uo["OU"] = np.asarray(todays_games_uo)
        ou_data = frame_uo.values.astype(float)
        ou_preds = np.array(xgb_uo.predict(xgb.DMatrix(ou_data))).reshape(len(games), -1)
    except Exception as e:
        raise PipelineError(f"Prediction failed: {e}")

    results = []
    history = []
    current_bankroll = bankroll

    for i, (home_team, away_team) in enumerate(games):
        # Winner prediction
        winner_idx = int(np.argmax(ml_preds[i]))
        winner_conf = ml_preds[i][winner_idx] * 100

        # OU prediction
        ou_idx = int(np.argmax(ou_preds[i]))
        ou_conf = ou_preds[i][ou_idx] * 100

        # EV and Kelly
        ev_home = ev_away = None
        kelly_home = kelly_away = None
        if home_team_odds[i] and home_team_odds[i] > 0:
            ev_home = expected_value(ml_preds[i][1], float(home_team_odds[i]))
            if use_kelly:
                kelly_home = kelly_fraction(ml_preds[i][1], float(home_team_odds[i]))
        if away_team_odds[i] and away_team_odds[i] > 0:
            ev_away = expected_value(ml_preds[i][0], float(away_team_odds[i]))
            if use_kelly:
                kelly_away = kelly_fraction(ml_preds[i][0], float(away_team_odds[i]))

        # Bet sizing
        bet_size_home = current_bankroll * min(kelly_home or max_fraction, max_fraction) if use_kelly else current_bankroll * max_fraction
        bet_size_away = current_bankroll * min(kelly_away or max_fraction, max_fraction) if use_kelly else current_bankroll * max_fraction

        # Simulate outcome (Bernoulli trial)
        outcome_home = "WIN" if random.random() < ml_preds[i][1] else "LOSS"
        profit_home = bet_size_home * (home_team_odds[i] - 1) if outcome_home == "WIN" else -bet_size_home
        current_bankroll += profit_home
        history.append(current_bankroll)

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
            "bet_size_home": round(bet_size_home, 2),
            "bet_size_away": round(bet_size_away, 2),
            "outcome_home": outcome_home,
            "bankroll_after": round(current_bankroll, 2),
        }
        results.append(result)

        # Console output
        winner_color = Fore.GREEN if winner_idx == 1 else Fore.RED
        ou_color = Fore.MAGENTA if ou_idx == 0 else Fore.BLUE
        print(
            f"{winner_color}{home_team}{Style.RESET_ALL} vs "
            f"{Fore.RED if winner_idx == 1 else Fore.GREEN}{away_team}{Style.RESET_ALL} "
            f"{Fore.CYAN}({winner_conf:.1f}%){Style.RESET_ALL}: "
            f"{ou_color}{result['ou_prediction']}{Style.RESET_ALL} {todays_games_uo[i]} "
            f"{Fore.CYAN}({ou_conf:.1f}%){Style.RESET_ALL}"
        )
        print(
            f"{home_team} EV: {ev_home} | Kelly: {result['kelly_home']} | Outcome: {outcome_home} | Bankroll: {current_bankroll:.2f}"
        )

    # Metrics summary
    wins = sum(1 for r in results if r["outcome_home"] == "WIN")
    total_bets = len(results)
    win_rate = wins / total_bets if total_bets > 0 else 0
    metrics = {
        "final_bankroll": current_bankroll,
        "avg_EV": np.mean([r["ev_home"] for r in results if r["ev_home"] is not None]),
        "avg_Kelly_Bet": np.mean([r["kelly_home"] for r in results if r["kelly_home"] is not None]),
        "win_rate": win_rate,
        "total_bets": total_bets,
    }

    logger.info(f"Simulation completed | Final bankroll={metrics['final_bankroll']:.2f}, Win rate={metrics['win_rate']:.2%}")

    return results, history, metrics