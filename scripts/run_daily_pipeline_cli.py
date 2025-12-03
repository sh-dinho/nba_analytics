# ============================================================
# File: scripts/run_daily_pipeline_cli.py
# Purpose: Run daily NBA prediction pipeline end-to-end
# ============================================================

import argparse
import os
import subprocess
import datetime
import pandas as pd
from core.config import BASE_DATA_DIR, MODELS_DIR, RESULTS_DIR, SUMMARY_FILE
from core.log_config import setup_logger
from core.exceptions import PipelineError

# Ensure directories exist
os.makedirs(BASE_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

today = datetime.datetime.now().strftime("%Y%m%d")
LOG_FILE = os.path.join(RESULTS_DIR, f"pipeline_run_{today}.log")

logger = setup_logger("pipeline")


def ensure_player_stats(season="2024-25", force_refresh=False):
    args = ["python", "scripts/fetch_player_stats.py", "--season", season]
    if force_refresh:
        args.append("--force_refresh")
    logger.info(f"Ensuring player stats for season {season} (force_refresh={force_refresh})...")
    subprocess.run(args, check=True)


def main(threshold=0.6, strategy="kelly", max_fraction=0.05,
         season="2024-25", force_refresh=False, rounds=10):
    logger.info(f"Starting pipeline | threshold={threshold}, strategy={strategy}, "
                f"max_fraction={max_fraction}, season={season}, force_refresh={force_refresh}, rounds={rounds}")

    try:
        # Step 1: Ensure stats
        ensure_player_stats(season=season, force_refresh=force_refresh)

        # Step 2: Build features
        subprocess.run(["python", "scripts/build_features.py", "--rounds", str(rounds)], check=True)

        # Step 3: Train model
        subprocess.run(["python", "scripts/train_model.py"], check=True)

        # Step 4: Generate predictions
        subprocess.run([
            "python", "scripts/generate_today_predictions.py",
            "--threshold", str(threshold),
            "--strategy", strategy,
            "--max_fraction", str(max_fraction)
        ], check=True)

        # Step 5: Generate picks
        subprocess.run(["python", "scripts/generate_picks.py"], check=True)

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Pipeline step failed: {e}")
        raise PipelineError(f"Pipeline execution failed: {e}")

    # Step 6: Collect summaries
    try:
        n_games = len(pd.read_csv(os.path.join(BASE_DATA_DIR, "training_features.csv")))
    except Exception:
        n_games = 0

    try:
        metrics_df = pd.read_csv(os.path.join(RESULTS_DIR, "training_metrics.csv"))
        last_metrics = metrics_df.iloc[-1].to_dict()
        acc = last_metrics.get("accuracy")
        auc = last_metrics.get("auc")
        logloss = last_metrics.get("log_loss")
        brier = last_metrics.get("brier")
    except Exception:
        acc, auc, logloss, brier = None, None, None, None

    try:
        preds_df = pd.read_csv(os.path.join(RESULTS_DIR, "today_predictions.csv"))
        n_preds = len(preds_df)
        n_bets = int(preds_df.get("bet_recommendation", pd.Series([0])).sum())
        avg_prob = preds_df.get("win_prob", pd.Series([0.0])).mean()
    except Exception:
        n_preds, n_bets, avg_prob = 0, 0, 0.0

    try:
        picks_df = pd.read_csv(os.path.join(RESULTS_DIR, "picks.csv"))
        home_picks = (picks_df["pick"] == "HOME").sum()
        away_picks = (picks_df["pick"] == "AWAY").sum()
        avg_ev = picks_df["ev"].mean() if "ev" in picks_df.columns else None
    except Exception:
        home_picks, away_picks, avg_ev = 0, 0, None

    # Log summaries
    feature_summary = f"FEATURE SUMMARY: Games built={n_games}"
    training_summary = (f"TRAINING SUMMARY: Accuracy={acc:.4f} LogLoss={logloss:.4f} "
                        f"Brier={brier:.4f} AUC={auc:.4f}" if acc is not None else "TRAINING SUMMARY: unavailable")
    prediction_summary = (f"PREDICTION SUMMARY: Predictions={n_preds}, Bets recommended={n_bets}, "
                          f"Avg win_prob={avg_prob:.3f}, Threshold={threshold}, Strategy={strategy}, MaxFraction={max_fraction}")
    picks_summary = (f"PICKS SUMMARY: HOME={home_picks}, AWAY={away_picks}, Avg EV={avg_ev:.3f if avg_ev is not None else 'N/A'}")

    logger.info(feature_summary)
    logger.info(training_summary)
    logger.info(prediction_summary)
    logger.info(picks_summary)
    logger.info("Pipeline completed successfully")

    # Append to rolling CSV summary
    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_entry = pd.DataFrame([{
        "timestamp": run_time,
        "games_built": n_games,
        "accuracy": acc,
        "log_loss": logloss,
        "brier": brier,
        "auc": auc,
        "predictions": n_preds,
        "bets_recommended": n_bets,
        "avg_win_prob": avg_prob,
        "home_picks": home_picks,
        "away_picks": away_picks,
        "avg_ev": avg_ev,
        "threshold": threshold,
        "strategy": strategy,
        "max_fraction": max_fraction
    }])

    if os.path.exists(SUMMARY_FILE):
        summary_entry.to_csv(SUMMARY_FILE, mode="a", header=False, index=False)
    else:
        summary_entry.to_csv(SUMMARY_FILE, index=False)

    logger.info(f"Pipeline summary appended to {SUMMARY_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run daily NBA prediction pipeline")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--strategy", type=str, default="kelly")
    parser.add_argument("--max_fraction", type=float, default=0.05)
    parser.add_argument("--season", type=str, default="2024-25")
    parser.add_argument("--force_refresh", action="store_true")
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()

    main(threshold=args.threshold, strategy=args.strategy,
         max_fraction=args.max_fraction, season=args.season,
         force_refresh=args.force_refresh, rounds=args.rounds)