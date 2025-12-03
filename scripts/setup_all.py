# ============================================================
# File: scripts/setup_all.py
# Purpose: Full NBA analytics pipeline orchestration with safe execution, logging, and notifications
# ============================================================

from pathlib import Path
import json
from datetime import datetime
import pandas as pd

from core.config import (
    ensure_dirs,
    BASE_DATA_DIR,
    BASE_RESULTS_DIR,
    BASE_MODELS_DIR,
    TRAINING_FEATURES_FILE,
    MODEL_FILE_PKL,
)
from core.exceptions import PipelineError, DataError
from core.log_config import setup_logger

# Pipeline steps
from scripts.build_features import main as build_features
from app.train_model import main as train_model
from scripts.generate_today_predictions import generate_today_predictions   # ✅ corrected import
from scripts.generate_picks import main as generate_picks

# Optional utilities
try:
    from scripts.utils import ensure_columns, get_timestamp
except ImportError:
    def ensure_columns(df, required, label):
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {label}: {missing}")

    def get_timestamp():
        return datetime.now().strftime("%Y%m%d_%H%M%S")

# Optional notifications
try:
    from scripts.notifications import send_telegram_message, send_ev_summary
    HAS_NOTIFICATIONS = True
except ImportError:
    HAS_NOTIFICATIONS = False

logger = setup_logger("setup_all")
REQUIRED_PRED_COLS = {"game_id", "win_prob", "decimal_odds", "ev"}
_duplicate_paths: list[Path] = []


def _safe_run(step_name: str, func, *args, **kwargs):
    logger.info(f"===== Starting: {step_name} =====")
    try:
        out = func(*args, **kwargs)
        logger.info(f"✅ Completed: {step_name}")
        return out
    except Exception as e:
        logger.error(f"❌ {step_name} failed: {e}")
        raise PipelineError(f"{step_name} failed: {e}")


def _timestamped_copy(path: Path) -> Path:
    if path.exists():
        ts = get_timestamp()
        stamped = path.with_name(f"{path.stem}_{ts}{path.suffix}")
        _duplicate_paths.append(stamped)
        return stamped
    return path


def _cleanup_duplicates():
    for p in _duplicate_paths:
        try:
            if p.exists():
                p.unlink()
                logger.info(f"🗑️ Deleted duplicate file: {p}")
        except Exception as e:
            logger.warning(f"Failed to delete {p}: {e}")


def main(skip_train: bool = False, skip_fetch: bool = True, notify: bool = False, threshold: float = 0.6):
    try:
        ensure_dirs()
        BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        BASE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if not skip_fetch:
            logger.info("📡 Fetching live data (implement your fetch logic here)...")

        # 1) Build features
        _safe_run("Build Features", build_features)

        # 2) Train model
        metrics = {}
        if not skip_train:
            metrics = _safe_run("Train Model", train_model) or {}
            if isinstance(metrics, dict) and metrics:
                logger.info("=== TRAINING METRICS ===")
                for k in ("accuracy", "log_loss", "brier", "auc"):
                    if k in metrics and metrics[k] is not None:
                        logger.info(f"{k.capitalize()}: {metrics[k]:.3f}")

        # 3) Generate predictions
        preds = _safe_run(
            "Generate Today's Predictions",
            generate_today_predictions,
            features_file=str(TRAINING_FEATURES_FILE),
            threshold=threshold,
        )
        if preds is None or not isinstance(preds, pd.DataFrame):
            raise DataError("Predictions step returned no data")

        ensure_columns(preds, REQUIRED_PRED_COLS, "predictions")

        preds_path = BASE_RESULTS_DIR / "predictions.csv"
        preds.to_csv(preds_path, index=False)
        stamped_preds = _timestamped_copy(preds_path)
        if stamped_preds != preds_path:
            preds.to_csv(stamped_preds, index=False)

        # --- Prediction metrics block ---
        logger.info("=== PREDICTION METRICS ===")
        if "label" in preds.columns and not preds["label"].isnull().all():
            logger.info("Metrics (accuracy, brier, auc) were logged above by generate_today_predictions.")
        else:
            avg_prob = preds["pred_home_win_prob"].mean()
            win_rate = preds["predicted_home_win"].mean()
            logger.info(f"Average predicted home win probability: {avg_prob:.3f}")
            logger.info(f"Predicted home win rate: {win_rate:.3f}")

        # --- Prediction summary table (Matchups) ---
        if {"game_id", "home_team", "away_team", "pred_home_win_prob", "predicted_home_win"} <= set(preds.columns):
            logger.info("=== PREDICTION SUMMARY (Matchups) ===")
            for _, row in preds.iterrows():
                gid = row["game_id"]
                home = row["home_team"]
                away = row["away_team"]
                prob = row["pred_home_win_prob"]
                pred = "Home Win" if row["predicted_home_win"] == 1 else "Home Loss"
                logger.info(f"Game {gid}: {home} vs {away} → {prob:.3f} ({pred})")

        # --- Highest-confidence pick ---
        if "pred_home_win_prob" in preds.columns and len(preds) > 0:
            best_row = preds.loc[preds["pred_home_win_prob"].idxmax()]
            gid = best_row.get("game_id", "N/A")
            home = best_row.get("home_team", "N/A")
            away = best_row.get("away_team", "N/A")
            prob = best_row["pred_home_win_prob"]
            pred = "Home Win" if best_row["predicted_home_win"] == 1 else "Home Loss"
            logger.info("=== HIGHEST-CONFIDENCE PICK ===")
            logger.info(f"Game {gid}: {home} vs {away} → {prob:.3f} ({pred})")

        # --- Lowest-confidence game (closest to 50/50) ---
        if "pred_home_win_prob" in preds.columns and len(preds) > 0:
            closest_idx = (preds["pred_home_win_prob"] - 0.5).abs().idxmin()
            low_row = preds.loc[closest_idx]
            gid = low_row.get("game_id", "N/A")
            home = low_row.get("home_team", "N/A")
            away = low_row.get("away_team", "N/A")
            prob = low_row["pred_home_win_prob"]
            pred = "Home Win" if low_row["predicted_home_win"] == 1 else "Home Loss"
            logger.info("=== LOWEST-CONFIDENCE GAME ===")
            logger.info(f"Game {gid}: {home} vs {away} → {prob:.3f} ({pred})")

        # --- Export predictions summary CSV ---
        summary_rows = []
        if {"game_id", "home_team", "away_team", "pred_home_win_prob", "predicted_home_win"} <= set(preds.columns):
            for _, row in preds.iterrows():
                summary_rows.append({
                    "game_id": row["game_id"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "pred_home_win_prob": row["pred_home_win_prob"],
                    "predicted_home_win": row["predicted_home_win"],
                    "highlight": ""
                })
        if "pred_home_win_prob" in preds.columns and len(preds) > 0:
            best_row = preds.loc[preds["pred_home_win_prob"].idxmax()]
            summary_rows.append({
                "game_id": best_row.get("game_id", "N/A"),
                "home_team": best_row.get("home_team", "N/A"),
                "away_team": best_row.get("away_team", "N/A"),
                "pred_home_win_prob": best_row["pred_home_win_prob"],
                "predicted_home_win": best_row["predicted_home_win"],
                "highlight": "Highest-confidence"
            })
            closest_idx = (preds["pred_home_win_prob"] - 0.5).abs().idxmin()
            low_row = preds.loc[closest_idx]
            summary_rows.append({
                "game_id": low_row.get("game_id", "N/A"),
                "home_team": low_row.get("home_team", "N/A"),
                "away_team": low_row.get("away_team", "N/A"),
                "pred_home_win_prob": low_row["pred_home_win_prob"],
                "predicted_home_win": low_row["predicted_home_win"],
                "highlight": "Lowest-confidence"
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_path = BASE_RESULTS_DIR / "predictions_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        stamped_summary = _timestamped_copy(summary_path)
        if stamped_summary != summary_path:
                    summary_df = pd.DataFrame(summary_rows)
        summary_path = BASE_RESULTS_DIR / "predictions_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        stamped_summary = _timestamped_copy(summary_path)
        if stamped_summary != summary_path:
            summary_df.to_csv(stamped_summary, index=False)

        logger.info(f"📊 Predictions summary saved to {summary_path}")

        # 4) Generate Picks
        _safe_run("Generate Picks", generate_picks)
        picks_path = BASE_RESULTS_DIR / "picks.csv"
        if not picks_path.exists():
            raise DataError("Expected picks.csv not found after generate_picks")

        picks = pd.read_csv(picks_path)

        if "pick" in picks.columns:
            summary = picks["pick"].value_counts().rename_axis("side").reset_index(name="count")
            summary_path = BASE_RESULTS_DIR / "picks_summary.csv"
            summary.to_csv(summary_path, index=False)
            stamped_summary = _timestamped_copy(summary_path)
            if stamped_summary != summary_path:
                summary.to_csv(stamped_summary, index=False)

        # 5) Notifications
        if notify and HAS_NOTIFICATIONS:
            try:
                msg = (
                    f"Pipeline Summary\n"
                    f"Predictions: {len(preds)} games\n"
                    f"Picks saved to {picks_path}"
                )
                send_telegram_message(msg)
                send_ev_summary(picks)
            except Exception as e:
                logger.warning(f"Failed to send Telegram notification: {e}")
        elif notify and not HAS_NOTIFICATIONS:
            logger.warning("Notifications requested, but scripts.notifications is not available.")

        # Metadata
        meta = {
            "generated_at": datetime.now().isoformat(),
            "skip_train": skip_train,
            "skip_fetch": skip_fetch,
            "notify": notify,
            "predictions_rows": int(len(preds)),
            "picks_rows": int(len(picks)),
            "training_metrics": metrics if metrics else {},
        }
        meta_path = BASE_RESULTS_DIR / "pipeline_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        _cleanup_duplicates()
        logger.info("🎉 Pipeline completed successfully")
        return metrics

    except (PipelineError, DataError) as e:
        logger.error(f"❌ Pipeline terminated due to error: {e}")
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run full NBA analytics pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip live data fetch")
    parser.add_argument("--notify", action="store_true", help="Send Telegram notifications")
    parser.add_argument("--threshold", type=float, default=0.6, help="Prediction threshold")
    args = parser.parse_args()

    main(skip_train=args.skip_train, skip_fetch=args.skip_fetch, notify=args.notify, threshold=args.threshold)