# ============================================================
# File: scripts/generate_today_predictions.py
# Purpose: Generate predictions for today's games using the
#          trained model, compute EV + picks, update bankroll,
#          and optionally send Telegram notifications.
# Author: <your name or org>
# Last Updated: 2025-02-21
#
# Notes:
# - Uses model artifact saved via train_model.py
# - Supports margin, label, and outcome-category targets
# - CI/CD friendly: deterministic output files & logs
# ============================================================

import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from core.paths import LOGS_DIR, ensure_dirs
from core.config import (
    MODEL_FILE_PKL,
    FEATURES_FILE,
    PREDICTIONS_FILE,
    PICKS_FILE,
    PICKS_BANKROLL_FILE,
    DEFAULT_BANKROLL,
    MAX_KELLY_FRACTION,
)
from core.log_config import init_global_logger
from core.utils import ensure_columns
from core.exceptions import DataError, PipelineError, FileError

# Notifications are optional (CI may not inject secrets)
try:
    from notifications import send_telegram_message, send_ev_summary
    TELEGRAM_AVAILABLE = True
except Exception:
    TELEGRAM_AVAILABLE = False

logger = init_global_logger()
PREDICTIONS_LOG = LOGS_DIR / "predictions.log"


# ============================================================
# Utility: Safe Telegram Send
# ============================================================

def _safe_telegram(msg: str):
    if TELEGRAM_AVAILABLE:
        try:
            send_telegram_message(msg)
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")


# ============================================================
# Load Model
# ============================================================

def load_model() -> tuple:
    """Load trained model artifact and return (pipeline, feature_cols, target)."""
    if not Path(MODEL_FILE_PKL).exists():
        raise FileError("Model file not found", file_path=str(MODEL_FILE_PKL))

    artifact = joblib.load(MODEL_FILE_PKL)

    pipeline = artifact["model"]
    feature_cols = artifact["features"]
    target = artifact.get("target", "label")

    logger.info(f"✅ Loaded trained model (target={target})")
    return pipeline, feature_cols, target


# ============================================================
# Bankroll Update
# ============================================================

def update_bankroll(picks_df: pd.DataFrame):
    """Append today's bankroll delta into PICKS_BANKROLL_FILE."""
    if picks_df.empty:
        return

    today = pd.Timestamp.today().date().isoformat()

    total_stake = picks_df["stake_amount"].sum()
    avg_ev = picks_df["ev"].mean()
    bankroll_change = (picks_df["ev"] * picks_df["stake_amount"]).sum()

    record = {
        "Date": today,
        "Total_Stake": round(total_stake, 2),
        "Avg_EV": round(avg_ev, 4),
        "Bankroll_Change": round(bankroll_change, 4),
    }

    if Path(PICKS_BANKROLL_FILE).exists():
        hist = pd.read_csv(PICKS_BANKROLL_FILE)
        hist = pd.concat([hist, pd.DataFrame([record])], ignore_index=True)
    else:
        hist = pd.DataFrame([record])

    hist.to_csv(PICKS_BANKROLL_FILE, index=False)
    logger.info(f"💰 Bankroll updated → {PICKS_BANKROLL_FILE}")

    # Telegram
    msg = (
        f"🏀 Bankroll Update ({today})\n"
        f"💰 Total Stake: {record['Total_Stake']}\n"
        f"📈 Avg EV: {record['Avg_EV']}\n"
        f"💵 Bankroll Change: {record['Bankroll_Change']:+}"
    )
    _safe_telegram(msg)


# ============================================================
# Prediction Computation
# ============================================================

def prepare_predictions(df: pd.DataFrame, pipeline, feature_cols, target, threshold):
    """Add prediction columns to df in-place."""
    X = df[feature_cols]

    if target == "label":
        probs = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, "predict_proba") else pipeline.predict(X)
        df["pred_home_win_prob"] = probs
        df["predicted_home_win"] = (probs >= threshold).astype(int)

    elif target == "margin":
        preds = pipeline.predict(X)
        df["predicted_margin"] = preds
        df["pred_home_win_prob"] = 1 / (1 + np.exp(-0.1 * preds))
        df["predicted_home_win"] = (df["pred_home_win_prob"] >= threshold).astype(int)

    elif target == "outcome_category":
        preds = pipeline.predict(X)
        df["predicted_outcome_category"] = preds
        if hasattr(pipeline, "predict_proba"):
            prob_df = pd.DataFrame(
                pipeline.predict_proba(X),
                columns=[f"prob_{c}" for c in pipeline.classes_]
            )
            df[prob_df.columns] = prob_df

    else:
        raise DataError(f"Unknown target type: {target}")

    return df


# ============================================================
# Picks Generator
# ============================================================

def generate_picks(df: pd.DataFrame) -> pd.DataFrame:
    """Compute stakes, EV, and recommended bets."""
    if "decimal_odds" not in df.columns or "pred_home_win_prob" not in df.columns:
        logger.info("No odds present; skipping pick generation.")
        return pd.DataFrame()

    picks = []
    for _, row in df.iterrows():
        odds = row["decimal_odds"]
        if pd.isna(odds) or odds <= 1:
            continue

        p = row["pred_home_win_prob"]
        b = odds - 1
        q = 1 - p

        ev = p * odds - 1
        kf = (b * p - q) / b if b > 0 else 0

        if kf > 0:
            kf = min(kf, MAX_KELLY_FRACTION)
            stake_amount = DEFAULT_BANKROLL * kf

            picks.append({
                "game_id": row["game_id"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "pred_home_win_prob": p,
                "predicted_home_win": row.get("predicted_home_win"),
                "decimal_odds": odds,
                "ev": ev,
                "kelly_fraction": kf,
                "stake_amount": stake_amount,
            })

    return pd.DataFrame(picks)


# ============================================================
# Save Files
# ============================================================

def save_predictions(df: pd.DataFrame):
    df.to_csv(PREDICTIONS_FILE, index=False)
    logger.info(f"📊 Predictions saved → {PREDICTIONS_FILE}")


def save_picks(picks_df: pd.DataFrame):
    picks_df.to_csv(PICKS_FILE, index=False)
    logger.info(f"💾 Picks saved → {PICKS_FILE}")


def append_prediction_summary(num_games, num_picks, threshold):
    entry = pd.DataFrame([{
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "games": num_games,
        "picks": num_picks,
        "threshold": threshold,
        "predictions_file": str(PREDICTIONS_FILE),
        "picks_file": str(PICKS_FILE) if num_picks > 0 else None,
    }])

    mode = "a" if PREDICTIONS_LOG.exists() else "w"
    header = not PREDICTIONS_LOG.exists()

    entry.to_csv(PREDICTIONS_LOG, mode=mode, header=header, index=False)
    logger.info(f"📈 Summary appended → {PREDICTIONS_LOG}")


# ============================================================
# Main Prediction Function
# ============================================================

def generate_today_predictions(features_file: str, threshold: float = 0.6) -> pd.DataFrame:
    ensure_dirs(strict=False)

    features_path = Path(features_file)
    if not features_path.exists():
        raise FileError("Features file not found", file_path=str(features_file))

    df = pd.read_csv(features_path)
    if df.empty:
        raise DataError("Features file is empty: no games today.")

    pipeline, feature_cols, target = load_model()

    ensure_columns(df, feature_cols + ["game_id", "home_team", "away_team"], "game features")

    df = prepare_predictions(df, pipeline, feature_cols, target, threshold)
    save_predictions(df)

    picks_df = generate_picks(df)

    if not picks_df.empty:
        save_picks(picks_df)
        update_bankroll(picks_df)

        # Telegram
        if TELEGRAM_AVAILABLE:
            try:
                send_ev_summary(picks_df)
            except Exception as e:
                logger.warning(f"send_ev_summary failed: {e}")
    else:
        logger.info("ℹ️ No positive EV picks today.")
        _safe_telegram("ℹ️ No positive EV picks found today.")

    append_prediction_summary(len(df), len(picks_df), threshold)

    # Final message
    msg = f"SUMMARY: Games={len(df)}, Picks={len(picks_df)}, Threshold={threshold}"
    _safe_telegram(msg)

    return df


# ============================================================
# Print Latest Summary
# ============================================================

def print_latest_summary():
    if not PREDICTIONS_LOG.exists():
        logger.error("No predictions summary log found.")
        return

    try:
        df = pd.read_csv(PREDICTIONS_LOG)
        if df.empty:
            logger.warning("Predictions summary log empty.")
            return

        latest = df.tail(1).iloc[0].to_dict()
        logger.info(f"📊 Latest summary → {latest}")
    except Exception as e:
        logger.error(f"Failed to read summary log: {e}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate today's predictions")
    parser.add_argument("--threshold", type=float, default=0.6, help="Win probability threshold")
    parser.add_argument("--summary-only", action="store_true", help="Print latest summary only")
    args = parser.parse_args()

    if args.summary_only:
        print_latest_summary()
    else:
        generate_today_predictions(FEATURES_FILE, threshold=args.threshold)
