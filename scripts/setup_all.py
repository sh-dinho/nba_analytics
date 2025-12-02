# File: scripts/setup_all.py
import os
import pandas as pd
import argparse
import sys
import logging
import json
from datetime import datetime

# Import pipeline steps
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


def _ensure_columns(df, required_cols, name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def main(skip_train=False, skip_fetch=True, notify=False):
    # Ensure directories exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 0) Fetch live data (optional)
    if not skip_fetch:
        logger.info("🛰️ Fetching live data... (implement fetch scripts here)")

    # 1) Build features
    logger.info("🛠️ Building features...")
    try:
        build_features()
    except Exception as e:
        logger.error(f"❌ Error building features: {e}")
        return None

    # 2) Train model
    metrics = None
    if not skip_train:
        logger.info("🤖 Training model with calibration + SHAP...")
        try:
            metrics = train_model()  # returns metrics dict
        except ValueError:
            logger.info("ℹ️ No outcomes available — skipping training, using existing model if present.")
        except Exception as e:
            logger.error(f"❌ Error training model: {e}")

        if metrics:
            logger.info("\n=== TRAINING METRICS ===")
            logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
            logger.info(f"Log Loss: {metrics['log_loss']:.3f}")
            logger.info(f"Brier Score: {metrics['brier']:.3f}")
            logger.info(f"AUC: {metrics['auc']:.3f}")
            logger.info("📈 Calibration curve saved to models/calibration_curve.png")
            logger.info("📈 SHAP feature importance saved to models/shap_feature_importance.png")

    # 3) Generate predictions
    logger.info("🔮 Generating predictions...")
    preds = None
    try:
        preds = generate_today_predictions()
        _ensure_columns(preds, REQUIRED_PRED_COLS, "predictions.csv")
        preds_file = "results/predictions.csv"
        preds.to_csv(preds_file, index=False)
        ts_preds_file = preds_file.replace(".csv", f"_{_timestamp()}.csv")
        preds.to_csv(ts_preds_file, index=False)
        logger.info(f"✅ Predictions saved to {preds_file} ({len(preds)} rows)")
        logger.info(f"📦 Timestamped backup saved to {ts_preds_file}")
    except Exception as e:
        logger.error(f"❌ Error generating predictions: {e}")
        return metrics

    # 4) Generate picks
    logger.info("🏀 Generating picks...")
    picks = None
    try:
        generate_picks()
        logger.info("✅ Picks saved to results/picks.csv")
        # Save summary
        picks = pd.read_csv("results/picks.csv")
        if "pick" in picks.columns:
            summary = picks["pick"].value_counts().reset_index()
            summary.columns = ["side", "count"]
            summary.to_csv("results/picks_summary.csv", index=False)
            ts_summary_file = f"results/picks_summary_{_timestamp()}.csv"
            summary.to_csv(ts_summary_file, index=False)
            logger.info("📊 Picks summary saved to results/picks_summary.csv")
            logger.info(f"📦 Timestamped backup saved to {ts_summary_file}")
            home_count = summary.loc[summary["side"] == "HOME", "count"].sum()
            away_count = summary.loc[summary["side"] == "AWAY", "count"].sum()
            logger.info(f"🔎 HOME picks: {home_count}")
            logger.info(f"🔎 AWAY picks: {away_count}")
    except Exception as e:
        logger.error(f"❌ Error generating picks: {e}")

    # 5) Notifications
    if notify and preds is not None and picks is not None:
        try:
            msg = (
                f"📊 Pipeline Summary\n"
                f"Predictions: {len(preds)} games\n"
                f"Picks saved to results/picks.csv"
            )
            send_telegram_message(msg)
            send_ev_summary(picks)
            logger.info("✅ Telegram notification sent")
        except Exception as e:
            logger.warning(f"⚠️ Failed to send Telegram notification: {e}")

    # Save metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "skip_train": skip_train,
        "skip_fetch": skip_fetch,
        "notify": notify,
        "predictions_rows": len(preds) if preds is not None else 0,
        "picks_rows": len(picks) if picks is not None else 0,
        "metrics": metrics if metrics else {}
    }
    meta_file = "results/pipeline_meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"🧾 Pipeline metadata saved to {meta_file}")

    return metrics  # <-- return metrics for CLI integration


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