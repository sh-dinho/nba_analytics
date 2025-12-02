# File: scripts/setup_all.py

import os
import pandas as pd
import argparse
import sys
import logging
import json
from datetime import datetime

# Stage imports
from scripts.build_features import main as build_features
from scripts.train_model import main as train_model
from app.predict_pipeline import generate_today_predictions
from scripts.generate_picks import main as generate_picks
from nba_analytics_core.notifications import send_telegram_message, send_ev_summary

REQUIRED_PRED_COLS = {"game_id", "home_win_prob"}

# ----------------------------
# Logging setup
# ----------------------------
logger = logging.getLogger("setup_all")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)


# ===============================================================
# Helpers
# ===============================================================

def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _ensure_columns(df: pd.DataFrame, required_cols, name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _safe_run(step_name: str, func, *args, **kwargs):
    """
    Standardized wrapper for each pipeline step.
    Handles logging, errors, and returns (result, success_flag).
    """
    logger.info(f"\n===== 🚀 Starting: {step_name} =====")

    try:
        out = func(*args, **kwargs)
        logger.info(f"✅ Completed: {step_name}")
        return out, True
    except Exception as e:
        logger.error(f"❌ {step_name} failed: {e}")
        return None, False


# ===============================================================
# Pipeline Orchestrator
# ===============================================================

def main(skip_train=False, skip_fetch=True, notify=False):
    # Ensure necessary directories
    for d in ["results", "data", "models"]:
        os.makedirs(d, exist_ok=True)

    # ===========================
    # 0) Fetch live data (optional)
    # ===========================
    if not skip_fetch:
        logger.info("🛰️ Fetching live data (placeholder — implement your actual fetch logic)...")

    # ===========================
    # 1) Build features
    # ===========================
    _, ok_features = _safe_run("Build Features", build_features)
    if not ok_features:
        logger.error("Stopping pipeline: cannot build features.")
        return None

    # ===========================
    # 2) Train Model
    # ===========================
    metrics = {}
    if not skip_train:
        metrics, ok_train = _safe_run("Train Model", train_model)
        if ok_train and metrics:
            logger.info("\n=== TRAINING METRICS ===")
            for k in ["accuracy", "log_loss", "brier", "auc"]:
                if k in metrics:
                    logger.info(f"{k.capitalize()}: {metrics[k]:.3f}")
            logger.info("📈 Calibration curve saved to models/calibration_curve.png")
            logger.info("📈 SHAP feature importance saved to models/shap_feature_importance.png")
        else:
            logger.info("⚠ Training skipped or failed; using existing model (if available).")

    # ===========================
    # 3) Generate Predictions
    # ===========================
    preds, ok_preds = _safe_run("Generate Today Predictions", generate_today_predictions)
    if not ok_preds or preds is None:
        logger.error("Stopping pipeline: cannot generate predictions.")
        return metrics

    try:
        _ensure_columns(preds, REQUIRED_PRED_COLS, "predictions")
    except Exception as e:
        logger.error(f"❌ Invalid predictions: {e}")
        return metrics

    preds_file = "results/predictions.csv"
    preds.to_csv(preds_file, index=False)
    preds_ts = preds_file.replace(".csv", f"_{_timestamp()}.csv")
    preds.to_csv(preds_ts, index=False)

    logger.info(f"✅ Predictions saved to {preds_file} ({len(preds)} rows)")
    logger.info(f"📦 Timestamped backup saved to {preds_ts}")

    # ===========================
    # 4) Generate Picks
    # ===========================
    _, ok_picks = _safe_run("Generate Picks", generate_picks)
    if not ok_picks:
        logger.error("Stopping pipeline: generate_picks failed.")
        return metrics

    picks_path = "results/picks.csv"
    if not os.path.exists(picks_path):
        logger.error("❌ Expected picks.csv not found after generate_picks.")
        return metrics

    picks = pd.read_csv(picks_path)

    # Export summary
    if "pick" in picks.columns:
        summary = picks["pick"].value_counts().reset_index()
        summary.columns = ["side", "count"]
        summary_file = "results/picks_summary.csv"
        summary.to_csv(summary_file, index=False)
        summary_ts = f"results/picks_summary_{_timestamp()}.csv"
        summary.to_csv(summary_ts, index=False)
        logger.info(f"📊 Picks summary saved to {summary_file}")
        logger.info(f"📦 Timestamped backup saved to {summary_ts}")

        logger.info(f"🔎 HOME picks: {summary.loc[summary['side']=='HOME','count'].sum()}")
        logger.info(f"🔎 AWAY picks: {summary.loc[summary['side']=='AWAY','count'].sum()}")

    # ===========================
    # 5) Send Notifications
    # ===========================
    if notify:
        try:
            msg = (
                f"📊 Pipeline Summary\n"
                f"Predictions: {len(preds)} games\n"
                f"Picks saved to results/picks.csv"
            )
            send_telegram_message(msg)
            send_ev_summary(picks)
            logger.info("📨 Telegram notification sent")
        except Exception as e:
            logger.warning(f"⚠️ Failed to send Telegram notification: {e}")

    # ===========================
    # Save pipeline metadata
    # ===========================
    meta = {
        "generated_at": datetime.now().isoformat(),
        "skip_train": skip_train,
        "skip_fetch": skip_fetch,
        "notify": notify,
        "predictions_rows": len(preds),
        "picks_rows": len(picks),
        "training_metrics": metrics if metrics else {},
    }

    meta_path = "results/pipeline_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"🧾 Pipeline metadata saved to {meta_path}")
    logger.info("🎉 Pipeline completed successfully")

    return metrics


# ===============================================================
# CLI
# ===============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full NBA analytics pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip live data fetch")
    parser.add_argument("--notify", action="store_true", help="Send Telegram notification after run")
    args = parser.parse_args()

    try:
        main(skip_train=args.skip_train, skip_fetch=args.skip_fetch, notify=args.notify)
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        sys.exit(1)
