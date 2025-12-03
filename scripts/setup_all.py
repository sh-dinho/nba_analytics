# ============================================================
# File: scripts/setup_all.py
# Purpose: Orchestrate full NBA analytics pipeline
# ============================================================

import argparse
import os
import pandas as pd
from pathlib import Path

from core.config import (
    BASE_RESULTS_DIR, DEFAULT_THRESHOLD, DEFAULT_BANKROLL, MAX_KELLY_FRACTION,
    HISTORICAL_GAMES_FILE
)
from core.log_config import setup_logger
from core.exceptions import PipelineError
from scripts.utils import ensure_columns, get_timestamp, log_positive_ev, Simulation

# Import pipeline steps
from scripts.fetch_new_games import fetch_new_games
from scripts.build_features import build_features_for_new_games
from scripts.build_features_for_training import build_features_for_training
from scripts.generate_synthetic_historical_games import generate_synthetic_historical_games
from app.train_model import main as train_model
from scripts.generate_today_predictions import generate_today_predictions

logger = setup_logger("setup_all")

REQUIRED_PRED_COLS = [
    "game_id", "home_team", "away_team",
    "pred_home_win_prob", "predicted_home_win", "decimal_odds"
]


def _safe_run(step_name, func, *args, **kwargs):
    logger.info(f"===== Starting: {step_name} =====")
    try:
        result = func(*args, **kwargs)
        logger.info(f"✅ Completed: {step_name}")
        return result
    except Exception as e:
        logger.error(f"❌ {step_name} failed: {e}")
        raise PipelineError(f"{step_name} failed: {e}")


def _timestamped_copy(path: Path) -> Path:
    ts = get_timestamp()
    new_path = path.with_name(f"{path.stem}_{ts}{path.suffix}")
    try:
        os.replace(path, new_path)
    except Exception:
        pass
    return new_path


def parse_args():
    parser = argparse.ArgumentParser(description="NBA Analytics Pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching new games")
    parser.add_argument("--notify", action="store_true", help="Send notifications")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Prediction threshold")
    parser.add_argument("--simulate", action="store_true", help="Run bankroll simulation on positive EV picks")
    return parser.parse_args()


def main(skip_train=False, skip_fetch=False, notify=False, threshold=DEFAULT_THRESHOLD, simulate=False):
    # Step 1: Fetch new games (for prediction)
    new_games_file = None
    if not skip_fetch:
        new_games_file = _safe_run("Fetch New Games", fetch_new_games)

    # Step 2: Build features
    if skip_train:
        # Prediction path → use new games
        features_file = _safe_run("Build Features (Prediction)", build_features_for_new_games, new_games_file)
    else:
        # Training path → use historical games
        if not os.path.exists(HISTORICAL_GAMES_FILE):
            logger.warning(f"{HISTORICAL_GAMES_FILE} not found. Generating synthetic historical data...")
            _safe_run("Generate Synthetic Historical Games", generate_synthetic_historical_games)
        features_file = _safe_run("Build Features (Training)", build_features_for_training)

    # Step 3: Train model (only if not skipped)
    if not skip_train:
        metrics = _safe_run("Train Model", train_model)
        logger.info("=== TRAINING METRICS ===")
        for k, v in metrics.items():
            logger.info(f"{k.capitalize()}: {v:.3f}" if v is not None else f"{k.capitalize()}: unavailable")

    # Step 4: Generate predictions (always runs)
    preds = _safe_run("Generate Today's Predictions", generate_today_predictions,
                      features_file=features_file, threshold=threshold)

    # Step 5: Ensure required columns
    ensure_columns(preds, REQUIRED_PRED_COLS, "predictions")

    # Step 6: Export predictions summary with EV for both sides
    summary_rows = []
    for _, row in preds.iterrows():
        prob_home_win = row["pred_home_win_prob"]
        odds = row.get("decimal_odds", 1.0)

        ev_home = prob_home_win * (odds - 1) - (1 - prob_home_win)
        prob_away_win = 1 - prob_home_win
        ev_away = prob_away_win * (odds - 1) - (1 - prob_away_win)

        summary_rows.append({
            "game_id": row.get("game_id", "N/A"),
            "home_team": row.get("home_team", "N/A"),
            "away_team": row.get("away_team", "N/A"),
            "pred_home_win_prob": prob_home_win,
            "predicted_home_win": row["predicted_home_win"],
            "decimal_odds": odds,
            "ev_home": ev_home,
            "ev_away": ev_away,
            "highlight": "HOME+" if ev_home > 0 else ("AWAY+" if ev_away > 0 else "")
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = BASE_RESULTS_DIR / "predictions_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    stamped_summary = _timestamped_copy(summary_path)
    if stamped_summary != summary_path:
        summary_df.to_csv(stamped_summary, index=False)

    logger.info(f"📊 Predictions summary with EV (home & away) saved to {summary_path}")

    # Step 7: Positive EV logging + separate CSVs + optional simulation
    log_positive_ev(summary_df)

    positive_ev_home = summary_df[summary_df["ev_home"] > 0]
    positive_ev_away = summary_df[summary_df["ev_away"] > 0]

    if not positive_ev_home.empty:
        pos_ev_home_path = BASE_RESULTS_DIR / f"positive_ev_home_{get_timestamp()}.csv"
        positive_ev_home.to_csv(pos_ev_home_path, index=False)
        logger.info(f"💡 Positive EV HOME picks saved to {pos_ev_home_path}")

    if not positive_ev_away.empty:
        pos_ev_away_path = BASE_RESULTS_DIR / f"positive_ev_away_{get_timestamp()}.csv"
        positive_ev_away.to_csv(pos_ev_away_path, index=False)
        logger.info(f"💡 Positive EV AWAY picks saved to {pos_ev_away_path}")

    if simulate and (not positive_ev_home.empty or not positive_ev_away.empty):
        bets = []
        for _, row in positive_ev_home.iterrows():
            bets.append({"prob_win": row["pred_home_win_prob"], "odds": row["decimal_odds"], "side": "home"})
        for _, row in positive_ev_away.iterrows():
            bets.append({"prob_win": 1 - row["pred_home_win_prob"], "odds": row["decimal_odds"], "side": "away"})

        sim = Simulation(initial_bankroll=DEFAULT_BANKROLL)
        sim.run(bets, strategy="kelly", max_fraction=MAX_KELLY_FRACTION)
        sim_summary = sim.summary()
        logger.info("=== POSITIVE EV SIMULATION SUMMARY ===")
        for k, v in sim_summary.items():
            logger.info(f"{k}: {v}")
    else:
        logger.info("No positive EV picks to save or simulate.")


if __name__ == "__main__":
    args = parse_args()
    main(skip_train=args.skip_train, skip_fetch=args.skip_fetch,
         notify=args.notify, threshold=args.threshold, simulate=args.simulate)