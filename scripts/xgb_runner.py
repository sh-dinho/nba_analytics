# ============================================================
# File: scripts/xgb_runner.py
# XGBoost Runner – loads ML & OU models, predicts outcomes,
# calculates EV & Kelly Criterion, prints colored output, returns structured results
# ============================================================

import copy
import numpy as np
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from scripts.utils import expected_value, kelly_fraction
from core.log_config import setup_logger
from core.exceptions import PipelineError

logger = setup_logger("xgb_runner")

# Initialize colorama
init()

# Load models once
try:
    xgb_ml = xgb.Booster()
    xgb_ml.load_model("Models/XGBoost_Models/XGBoost_68.7%_ML-4.json")
    logger.info("✅ Moneyline XGBoost model loaded")

    xgb_uo = xgb.Booster()
    xgb_uo.load_model("Models/XGBoost_Models/XGBoost_53.7%_UO-9.json")
    logger.info("✅ Over/Under XGBoost model loaded")
except Exception as e:
    logger.error(f"❌ Failed to load XGBoost models: {e}")
    raise PipelineError(f"Model loading failed: {e}")


def xgb_runner(data, todays_games_uo, frame_ml, games,
               home_team_odds, away_team_odds, use_kelly=True):
    """
    Run XGBoost models for moneyline and over/under predictions.
    Returns structured results for downstream use.
    """
    try:
        # Batch ML predictions
        ml_preds = xgb_ml.predict(xgb.DMatrix(np.array(data)))

        # Prepare OU data
        frame_uo = copy.deepcopy(frame_ml)
        frame_uo["OU"] = np.asarray(todays_games_uo)
        ou_data = frame_uo.values.astype(float)
        ou_preds = xgb_uo.predict(xgb.DMatrix(ou_data))
    except Exception as e:
        raise PipelineError(f"Prediction failed: {e}")

    results = []
    for i, (home_team, away_team) in enumerate(games):
        # Winner prediction
        winner_idx = int(np.argmax(ml_preds[i]))
        winner_conf = ml_preds[i][winner_idx] * 100 if hasattr(ml_preds[i], "__getitem__") else ml_preds[i] * 100

        # OU prediction
        ou_idx = int(np.argmax(ou_preds[i]))
        ou_conf = ou_preds[i][ou_idx] * 100 if hasattr(ou_preds[i], "__getitem__") else ou_preds[i] * 100

        # Expected values
        ev_home = ev_away = None
        if home_team_odds[i] and away_team_odds[i]:
            ev_home = expected_value(ml_preds[i][1], float(home_team_odds[i]))
            ev_away = expected_value(ml_preds[i][0], float(away_team_odds[i]))

        # Kelly fractions
        kelly_home = kelly_fraction(ml_preds[i][1], float(home_team_odds[i])) if use_kelly else None
        kelly_away = kelly_fraction(ml_preds[i][0], float(away_team_odds[i])) if use_kelly else None

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

        # Console output
        winner_color = Fore.GREEN if winner_idx == 1 else Fore.RED
        ou_color = Fore.MAGENTA if ou_idx == 0 else Fore.BLUE
        print(
            f"{winner_color}{home_team}{Style.RESET_ALL} vs "
            f"{Fore.RED if winner_idx == 1 else Fore.GREEN}{away_team}{Style.RESET_ALL} "
            f"{Fore.CYAN}({round(winner_conf,1)}%){Style.RESET_ALL}: "
            f"{ou_color}{result['ou_prediction']} {Style.RESET_ALL}{todays_games_uo[i]} "
            f"{Fore.CYAN}({round(ou_conf,1)}%){Style.RESET_ALL}"
        )

        ev_home_color = Fore.GREEN if ev_home and ev_home > 0 else Fore.RED
        ev_away_color = Fore.GREEN if ev_away and ev_away > 0 else Fore.RED
        print(
            f"{home_team} EV: {ev_home_color}{ev_home}{Style.RESET_ALL}" +
            (f" Fraction of Bankroll: {result['kelly_home']}%" if kelly_home else "")
        )
        print(
            f"{away_team} EV: {ev_away_color}{ev_away}{Style.RESET_ALL}" +
            (f" Fraction of Bankroll: {result['kelly_away']}%" if kelly_away else "")
        )

    deinit()
    return results