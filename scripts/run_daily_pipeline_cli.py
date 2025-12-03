# ============================================================
# File: scripts/run_daily_pipeline_cli.py
# Purpose: Run daily NBA prediction pipeline end-to-end
# ============================================================

import argparse
import os
import subprocess
import datetime
import pandas as pd
from pathlib import Path
from core.config import ensure_dirs, BASE_DATA_DIR, MODELS_DIR, RESULTS_DIR, SUMMARY_FILE
from core.log_config import setup_logger
from core.exceptions import PipelineError

# Ensure directories exist using centralized config
ensure_dirs()

today = datetime.datetime.now().strftime("%Y%m%d")
LOG_FILE = os.path.join(RESULTS_DIR, f"pipeline_run_{today}.log")

logger = setup_logger("pipeline")


def get_current_season() -> str:
    """Return current NBA season string like '2025-26' based on today's date."""
    today = datetime.date.today()
    year = today.year
    month = today.month
    return f"{year}-{str(year+1)[-2:]}" if month >= 10 else f"{year-1}-{str(year)[-2:]}"


def run_step(args, step_name: str):
    """Run a subprocess step with logging and error handling."""
    logger.info(f"▶️ {step_name}...")
    try:
        subprocess.run(args, check=True, capture_output=True, text=True)
        logger.info(f"✅ {step_name} completed")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {step_name} failed: {e.stderr}")
        raise PipelineError(f"{step_name} failed: {e.stderr}")


def ensure_player_stats(season="2025-26", force_refresh=False):
    args = ["python", "-m", "scripts.fetch_player_stats", "--season", season]
    if force_refresh:
        args.append("--force_refresh")
    run_step(args, f"Ensuring player stats for season {season}")


def format_metric(name, value, fmt=".4f"):
    """Helper to safely format metrics that may be None."""
    return f"{name}={format(value, fmt)}" if value is not None else f"{name}=N/A"


def main(threshold=0.6, strategy="kelly", max_fraction=0.05,
         season=None, force_refresh=False, rounds=10,
         target="label", model_type="logistic") -> pd.DataFrame:

    season = season or get_current_season()
    logger.info(f"📅 Auto-detected NBA season: {season}")
    logger.info(f"Starting pipeline | threshold={threshold}, strategy={strategy}, "
                f"max_fraction={max_fraction}, season={season}, force_refresh={force_refresh}, "
                f"rounds={rounds}, target={target}, model_type={model_type}")

    # Steps 1–5
    ensure_player_stats(season=season, force_refresh=force_refresh)

    if target in ["label", "margin", "outcome_category"]:
        run_step(["python", "-m", "scripts.build_features", "--rounds", str(rounds), "--training"],
                 "Building training features")
    else:
        run_step(["python", "-m", "scripts.build_features", "--rounds", str(rounds)],
                 "Building new game features")

    run_step(["python", "-m", "scripts.train_model", "--target", target, "--model_type", model_type],
             "Training model")

    run_step(["python", "-m", "scripts.generate_today_predictions",
              "--threshold", str(threshold), "--strategy", strategy, "--max_fraction", str(max_fraction)],
             "Generating predictions")

    run_step(["python", "-m", "scripts.generate_picks"], "Generating picks")

    # Step 6: Collect summaries
    try:
        n_games = len(pd.read_csv(BASE_DATA_DIR / "training_features.csv"))
    except Exception:
        n_games = 0

    try:
        metrics_df = pd.read_csv(RESULTS_DIR / "training_metrics.csv")
        last_metrics = metrics_df.iloc[-1].to_dict()
        acc = last_metrics.get("accuracy")
        auc = last_metrics.get("auc")
        logloss = last_metrics.get("log_loss")
        brier = last_metrics.get("brier")
        rmse = last_metrics.get("rmse")
    except Exception:
        acc, auc, logloss, brier, rmse = None, None, None, None, None

    try:
        preds_df = pd.read_csv(RESULTS_DIR / "today_predictions.csv")
        n_preds = len(preds_df)
        n_bets = int(preds_df.get("bet_recommendation", pd.Series([0])).sum())
        avg_prob = preds_df.get("win_prob", pd.Series([0.0])).mean()
    except Exception:
        n_preds, n_bets, avg_prob = 0, 0, 0.0

    try:
        picks_df = pd.read_csv(RESULTS_DIR / "picks.csv")
        home_picks = (picks_df["pick"] == "HOME").sum() if "pick" in picks_df.columns else 0
        away_picks = (picks_df["pick"] == "AWAY").sum() if "pick" in picks_df.columns else 0
        avg_ev = picks_df["ev"].mean() if "ev" in picks_df.columns else None
        total_stake = picks_df["stake_amount"].sum() if "stake_amount" in picks_df.columns else 0.0
        expected_profit = (picks_df["ev"] * picks_df["stake_amount"]).sum() if "ev" in picks_df.columns else 0.0
    except Exception:
        home_picks, away_picks, avg_ev, total_stake, expected_profit = 0, 0, None, 0.0, 0.0

    # Log summaries
    logger.info(f"FEATURE SUMMARY: Games built={n_games}")
    logger.info("TRAINING SUMMARY: " + " ".join([
        format_metric("Accuracy", acc),
        format_metric("LogLoss", logloss),
        format_metric("Brier", brier),
        format_metric("AUC", auc),
        format_metric("RMSE", rmse)
    ]))
    logger.info(f"PREDICTION SUMMARY: Predictions={n_preds}, Bets recommended={n_bets}, "
                f"Avg win_prob={avg_prob:.3f}, Threshold={threshold}, Strategy={strategy}, "
                f"MaxFraction={max_fraction}, Target={target}, ModelType={model_type}")
    logger.info(f"PICKS SUMMARY: HOME={home_picks}, AWAY={away_picks}, "
                f"Avg EV={avg_ev:.3f if avg_ev is not None else 'N/A'}, "
                f"Total Stake={total_stake:.2f}, Expected Profit={expected_profit:.2f}")
    logger.info("Pipeline completed successfully")

    # Append to rolling CSV summary
    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_entry = pd.DataFrame([{
        "timestamp": run_time,
        "season": season,
        "games_built": n_games,
        "accuracy": acc,
        "log_loss": logloss,
        "brier": brier,
        "auc": auc,
        "rmse": rmse,
        "predictions": n_preds,
        "bets_recommended": n_bets,
        "avg_win_prob": avg_prob,
        "home_picks": home_picks,
        "away_picks": away_picks,
        "avg_ev": avg_ev,
        "total_stake": total_stake,
        "expected_profit": expected_profit,
        "threshold": threshold,
        "strategy": strategy,
        "max_fraction": max_fraction,
        "target": target,
        "model_type": model_type
    }])

    try:
        if Path(SUMMARY_FILE).exists():
            summary_entry.to_csv(SUMMARY_FILE, mode="a", header=False, index=False)
        else:
            summary_entry.to_csv(SUMMARY_FILE, index=False)
        logger.info(f"Pipeline summary appended to {SUMMARY_FILE}")
    except Exception as e:
        raise PipelineError(f"Failed to append pipeline summary: {e}")

    return summary_entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run daily NBA prediction pipeline")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--strategy", type=str, default="kelly")
    parser.add_argument("--max_fraction", type=float, default=0.05)
    parser.add_argument("--season", type=str, default=get_current_season(),
                        help="NBA season, auto-detected by default")
    parser.add_argument("--force_refresh", action="store_true")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--target", type=str, default="label",
                        help="Target column: label, margin, outcome_category")
    parser.add_argument("--model_type", type=str, default="logistic",
                        help="Model type: logistic, rf, linear")
    args = parser.parse_args()

    main(threshold=args.threshold,
            strategy=args.strategy,
            max_fraction=args.max_fraction,
            season=args.season,
            force_refresh=args.force_refresh,
            rounds=args.rounds,
            target=args.target,
            model_type=args.model_type)