# ============================================================
# File: scripts/generate_today_predictions.py
# Purpose: Generate today's predictions from trained model + notify Telegram
# ============================================================

import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from core.paths import DATA_DIR, LOGS_DIR, ensure_dirs
from core.config import (
    MODEL_FILE_PKL,
    PREDICTIONS_FILE,
    DEFAULT_BANKROLL,
    MAX_KELLY_FRACTION,
    PICKS_FILE,
    FEATURES_FILE,
    PICKS_BANKROLL_FILE,
)
from core.log_config import init_global_logger
from core.utils import ensure_columns
from core.exceptions import DataError, PipelineError, FileError
from notifications import send_telegram_message, send_ev_summary  # ✅ Telegram hooks

logger = init_global_logger()
PREDICTIONS_LOG = LOGS_DIR / "predictions.log"


def update_bankroll(picks_df: pd.DataFrame):
    """Update bankroll tracking file with today's picks results."""
    if picks_df is None or picks_df.empty:
        return
    today = pd.Timestamp.today().date().isoformat()
    total_stake = picks_df["stake_amount"].sum()
    avg_ev = picks_df["ev"].mean()
    bankroll_change = (picks_df["ev"] * picks_df["stake_amount"]).sum()
    record = {
        "Date": today,
        "Total_Stake": total_stake,
        "Avg_EV": avg_ev,
        "Bankroll_Change": bankroll_change,
    }
    if Path(PICKS_BANKROLL_FILE).exists():
        hist = pd.read_csv(PICKS_BANKROLL_FILE)
        hist = pd.concat([hist, pd.DataFrame([record])], ignore_index=True)
    else:
        hist = pd.DataFrame([record])
    hist.to_csv(PICKS_BANKROLL_FILE, index=False)
    logger.info(f"💰 Bankroll updated → {PICKS_BANKROLL_FILE}")

    # ✅ Notify Telegram bankroll update
    msg = (
        f"🏀 Bankroll Update ({today})\n"
        f"💰 Total Stake: {total_stake:.2f}\n"
        f"📈 Avg EV: {avg_ev:.3f}\n"
        f"💵 Bankroll Change: {bankroll_change:+.2f}"
    )
    send_telegram_message(msg)


def generate_today_predictions(features_file: str, threshold: float = 0.6) -> pd.DataFrame:
    """
    Generate predictions for today's games using the trained model.
    Handles binary, regression, and multi-class targets.
    Saves predictions to PREDICTIONS_FILE and returns DataFrame.
    """
    ensure_dirs(strict=False)
    features_file = Path(features_file)

    if not features_file.exists():
        raise FileError("Features file not found", file_path=str(features_file))

    df = pd.read_csv(features_file)
    if df.empty:
        raise DataError(f"{features_file} is empty. No games to predict today.")

    # Load trained model artifact
    if not Path(MODEL_FILE_PKL).exists():
        raise FileError("Model file not found", file_path=str(MODEL_FILE_PKL))

    artifact = joblib.load(MODEL_FILE_PKL)
    pipeline = artifact["model"]
    feature_cols = artifact["features"]
    target = artifact.get("target", "label")

    logger.info(f"✅ Loaded model (target={target}) from {MODEL_FILE_PKL}")

    # Required columns
    required = feature_cols + ["game_id", "home_team", "away_team"]
    if "decimal_odds" in df.columns:
        required.append("decimal_odds")
    ensure_columns(df, required, "game features")

    X = df[feature_cols]

    # Predict depending on target type
    if target == "label":
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X)[:, 1]
        else:
            probs = pipeline.predict(X)
        preds = (probs >= threshold).astype(int)
        df["pred_home_win_prob"] = probs
        df["predicted_home_win"] = preds

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
            df = pd.concat([df, prob_df], axis=1)

    # Save predictions
    try:
        df.to_csv(PREDICTIONS_FILE, index=False)
        logger.info(f"📊 Predictions saved to {PREDICTIONS_FILE} ({len(df)} rows)")
    except Exception as e:
        raise PipelineError(f"Failed to save predictions: {e}")

    # Picks logic
    picks = []
    if "decimal_odds" in df.columns and "pred_home_win_prob" in df.columns:
        logger.info("=== GAME-LEVEL PREDICTIONS WITH PICKS & STAKING ===")
        for _, row in df.iterrows():
            odds = row["decimal_odds"]
            if pd.isna(odds) or odds <= 1:
                continue

            p = row["pred_home_win_prob"]
            q = 1 - p
            b = odds - 1
            ev = p * odds - 1
            kelly_fraction = (b * p - q) / b if b > 0 else 0

            if kelly_fraction > 0:
                kelly_fraction = min(kelly_fraction, MAX_KELLY_FRACTION)
                stake_amount = DEFAULT_BANKROLL * kelly_fraction
                picks.append({
                    "game_id": row["game_id"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "pred_home_win_prob": row.get("pred_home_win_prob"),
                    "predicted_home_win": row.get("predicted_home_win"),
                    "decimal_odds": odds,
                    "ev": ev,
                    "kelly_fraction": kelly_fraction,
                    "stake_amount": stake_amount,
                })
                logger.info(
                    f"{row['home_team']} vs {row['away_team']} → EV={ev:.3f} | ✅ Pick | Stake={stake_amount:.2f}"
                )

    # Save picks if any
    if picks:
        picks_df = pd.DataFrame(picks)
        try:
            picks_df.to_csv(PICKS_FILE, index=False)
            logger.info(f"💾 Picks saved to {PICKS_FILE} ({len(picks)} rows)")
        except Exception as e:
            raise PipelineError(f"Failed to save picks: {e}")

        # Update bankroll + notify Telegram
        update_bankroll(picks_df)

        # ✅ Send EV summary to Telegram
        send_ev_summary(picks_df)
    else:
        logger.info("ℹ️ No positive EV picks found today.")
        send_telegram_message("ℹ️ No positive EV picks found today.")

    # Summary log + Telegram
    summary_msg = f"SUMMARY: Games={len(df)}, Picks={len(picks)}, Threshold={threshold}"
    logger.info(summary_msg)
    send_telegram_message(summary_msg)

    # Append summary log
    summary_entry = pd.DataFrame([{
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "games": len(df),
        "picks": len(picks),
        "threshold": threshold,
        "predictions_file": str(PREDICTIONS_FILE),
        "picks_file": str(PICKS_FILE) if picks else None,
    }])
    try:
        if PREDICTIONS_LOG.exists():
            summary_entry.to_csv(PREDICTIONS_LOG, mode="a", header=False, index=False)
        else:
            summary_entry.to_csv(PREDICTIONS_LOG, index=False)
        logger.info(f"📈 Predictions summary appended to {PREDICTIONS_LOG}")
    except Exception as e:
        logger.warning(f"Failed to append predictions summary: {e}")

    return df


def print_latest_summary():
    """Print the latest predictions summary entry without regenerating predictions."""
    if not PREDICTIONS_LOG.exists():
        logger.error("No predictions summary log found.")
        return
    try:
        df = pd.read_csv(PREDICTIONS_LOG)
        if df.empty:
            logger.warning("Predictions summary log is empty.")
            return
        latest = df.tail(1).iloc[0].to_dict()
        logger.info(f"📊 Latest predictions summary: {latest}")
    except Exception as e:
        logger.error(f"Failed to read predictions summary log: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate today's predictions")
    parser.add_argument("--threshold", type=float, default=0.6, help="Win probability threshold")
    parser.add_argument("--summary-only", action="store_true", help="Print the latest predictions summary log entry")
    args = parser.parse_args()

    if args.summary_only:
        print