# File: scripts/train_and_backtest.py

import argparse
import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    roc_auc_score
)
from lightgbm import LGBMClassifier

from nba_analytics_core.data import (
    fetch_historical_games,
    build_team_stats,
    build_matchup_features
)
from nba_analytics_core.odds import fetch_odds
from scripts.simulate_bankroll import simulate_bankroll
from nba_analytics_core.notifications import send_telegram_message

DEFAULT_MODEL_PATH = "models/calibrated_lightgbm.pkl"
RESULTS_DIR = "results"

logger = logging.getLogger("train_backtest")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)

SEED = 42
np.random.seed(SEED)


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def add_timestamp(path: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}_{timestamp()}{ext}"


def is_valid_feature_vector(vec: Any) -> bool:
    try:
        arr = np.array(vec, dtype=float)
        return arr.ndim == 1 and np.isfinite(arr).all()
    except Exception:
        return False


def safe_fetch_odds(home_team: str, away_team: str) -> Optional[Dict[str, Any]]:
    """
    Fetch odds. Fail gracefully.
    """
    try:
        odds = fetch_odds(home_team=home_team, away_team=away_team)
        if not odds:
            return None
        home_odds = odds.get("home_odds")
        if home_odds is None or home_odds <= 1:
            return None
        return {"home_odds": float(home_odds), **{k: v for k, v in odds.items() if k != "home_odds"}}
    except Exception as e:
        logger.warning(f"Failed to fetch odds for {home_team} vs {away_team}: {e}")
        return None


# ============================================================
# Model Training (Upgraded: LightGBM + Isotonic Calibration)
# ============================================================

def train_model(
    seasons: List[str],
    model_path: str = DEFAULT_MODEL_PATH
) -> Dict[str, Any]:

    logger.info(f"Starting LightGBM training on seasons: {', '.join(seasons)}")

    X, y = [], []
    feature_lengths = set()
    skipped = 0

    # ----------------------------
    # Retrieve training data
    # ----------------------------
    for season in seasons:
        games = fetch_historical_games(season)
        team_stats = build_team_stats(games)

        for g in games:
            home = g.get("home_team")
            away = g.get("away_team")
            if not home or not away:
                skipped += 1
                continue

            feats = build_matchup_features(home, away, team_stats)
            if not is_valid_feature_vector(feats):
                skipped += 1
                continue

            feature_lengths.add(len(feats))
            X.append(feats)
            y.append(1 if g.get("home_win") else 0)

    if not X:
        raise ValueError("No valid training data found.")

    if len(feature_lengths) != 1:
        raise ValueError(f"Feature length mismatch detected: {feature_lengths}")

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    # ----------------------------
    # Train/val split for metrics
    # ----------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # ----------------------------
    # Base LightGBM (unfitted)
    # ----------------------------
    base_model = LGBMClassifier(
        n_estimators=450,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.1,
        random_state=SEED
    )

    # ----------------------------
    # Isotonic Calibration
    # ----------------------------
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    calibrated.fit(X_tr, y_tr)

    # ----------------------------
    # Validation metrics
    # ----------------------------
    proba = calibrated.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_val, preds)),
        "log_loss": float(log_loss(y_val, proba)),
        "brier": float(brier_score_loss(y_val, proba)),
        "auc": float(roc_auc_score(y_val, proba)),
    }

    logger.info(f"Validation metrics: {json.dumps(metrics, indent=2)}")
    logger.info(f"Skipped games during training: {skipped}")

    # ----------------------------
    # Save model + metadata
    # ----------------------------
    ensure_dir(os.path.dirname(model_path) or ".")
    metadata = {
        "trained_on_seasons": seasons,
        "model_type": "LightGBM + Isotonic Calibration",
        "feature_length": int(list(feature_lengths)[0]),
        "skipped_games": skipped,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }

    joblib.dump(
        {"model": calibrated, "metadata": metadata},
        model_path
    )
    logger.info(f"Model saved to {model_path}")

    return {"model": calibrated, "metadata": metadata}


# ============================================================
# Backtesting
# ============================================================

def backtest_model(
    seasons: List[str],
    strategy: str = "kelly",
    max_fraction: float = 0.05,
    export_summary: str = os.path.join(RESULTS_DIR, "backtest_summary.csv"),
    export_detailed: str = os.path.join(RESULTS_DIR, "backtest_detailed.csv"),
    export_bankroll_progression_dir: str = os.path.join(RESULTS_DIR, "bankroll_progression"),
    notify: bool = False,
    model_path: str = DEFAULT_MODEL_PATH
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    logger.info(f"Backtesting model on: {', '.join(seasons)}")

    # Load model bundle
    bundle = joblib.load(model_path)
    model = bundle["model"]
    metadata = bundle["metadata"]
    f_len = metadata["feature_length"]

    ensure_dir(RESULTS_DIR)
    ensure_dir(export_bankroll_progression_dir)

    summary_rows = []
    detailed_rows = []

    total_skipped_feats = 0
    total_skipped_odds = 0

    # ----------------------------
    # Loop through seasons
    # ----------------------------
    for season in seasons:
        games = fetch_historical_games(season)
        team_stats = build_team_stats(games)

        bets = []
        skipped_feats = 0
        skipped_odds = 0

        for g in games:
            home = g.get("home_team")
            away = g.get("away_team")
            home_win = g.get("home_win")
            game_id = g.get("game_id")

            if not home or not away:
                skipped_feats += 1
                continue

            feats = build_matchup_features(home, away, team_stats)
            if not is_valid_feature_vector(feats) or len(feats) != f_len:
                skipped_feats += 1
                continue

            prob = float(model.predict_proba([feats])[0, 1])

            odds = safe_fetch_odds(home, away)
            if not odds:
                skipped_odds += 1
                continue

            o = odds["home_odds"]
            implied = 1.0 / o
            edge = prob - implied
            ev = (prob * (o - 1)) - (1 - prob)

            bets.append({
                "season": season,
                "game_id": game_id,
                "home_team": home,
                "away_team": away,
                "decimal_odds": o,
                "prob": prob,
                "implied_prob": implied,
                "edge": edge,
                "ev": ev,
                "home_win": home_win,
            })

        total_skipped_feats += skipped_feats
        total_skipped_odds += skipped_odds

        if not bets:
            logger.warning(f"No valid bets for {season}.")
            continue

        df_bets = pd.DataFrame(bets)

        # ----------------------------
        # Run bankroll simulation
        # ----------------------------
        progression, metrics = simulate_bankroll(
            df_bets,
            strategy=strategy,
            max_fraction=max_fraction
        )

        progression_path = os.path.join(
            export_bankroll_progression_dir,
            f"bankroll_progression_{season}_{timestamp()}.csv"
        )
        progression.to_csv(progression_path, index=False)

        # ----------------------------
        # Predictive metrics
        # ----------------------------
        acc = accuracy_score(df_bets["home_win"], (df_bets["prob"] >= 0.5))
        ll = log_loss(df_bets["home_win"], df_bets["prob"])
        brier = brier_score_loss(df_bets["home_win"], df_bets["prob"])

        summary_rows.append({
            "season": season,
            "roi": metrics["roi"],
            "win_rate": metrics["win_rate"],
            "final_bankroll": metrics["final_bankroll"],
            "num_bets": len(df_bets),
            "accuracy": acc,
            "log_loss": ll,
            "brier": brier,
            "skipped_features": skipped_feats,
            "skipped_odds": skipped_odds,
        })

        detailed_rows.extend(bets)

    # ----------------------------
    # Export results
    # ----------------------------
    if not summary_rows:
        logger.warning("No backtest results produced.")
        return pd.DataFrame(), pd.DataFrame()

    df_summary = pd.DataFrame(summary_rows)
    df_detailed = pd.DataFrame(detailed_rows)

    export_summary_ts = add_timestamp(export_summary)
    export_detailed_ts = add_timestamp(export_detailed)

    df_summary.to_csv(export_summary_ts, index=False)
    df_detailed.to_csv(export_detailed_ts, index=False)

    logger.info(f"Summary saved to {export_summary_ts}")
    logger.info(f"Detailed results saved to {export_detailed_ts}")
    logger.info(f"Total skipped feats={total_skipped_feats}, skipped odds={total_skipped_odds}")

    # ----------------------------
    # Optional Telegram notification
    # ----------------------------
    if notify:
        avg_roi = df_summary["roi"].mean()
        msg = (
            "📊 Backtest Summary\n"
            f"Seasons: {', '.join(seasons)}\n"
            f"Avg ROI: {avg_roi:.4f}\n"
            f"Total bets: {df_summary['num_bets'].sum()}\n"
        )
        try:
            send_telegram_message(msg)
        except Exception as e:
            logger.warning(f"Failed to send Telegram message: {e}")

    return df_summary, df_detailed


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train + Backtest LightGBM Model")

    p.add_argument("--seasons", nargs="+", required=True)
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)

    p.add_argument("--backtest", action="store_true")
    p.add_argument("--train_only", action="store_true")

    p.add_argument("--export_summary", type=str, default=f"{RESULTS_DIR}/backtest_summary.csv")
    p.add_argument("--export_detailed", type=str, default=f"{RESULTS_DIR}/backtest_detailed.csv")
    p.add_argument("--bankroll_progression_dir", type=str, default=f"{RESULTS_DIR}/bankroll_progression")

    p.add_argument("--strategy", choices=["kelly", "flat"], default="kelly")
    p.add_argument("--max_fraction", type=float, default=0.05)

    p.add_argument("--notify", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    train_model(args.seasons, args.model_path)

    if args.backtest and not args.train_only:
        backtest_model(
            seasons=args.seasons,
            strategy=args.strategy,
            max_fraction=args.max_fraction,
            export_summary=args.export_summary,
            export_detailed=args.export_detailed,
            export_bankroll_progression_dir=args.bankroll_progression_dir,
            notify=args.notify,
            model_path=args.model_path
        )
    else:
        logger.info("Training completed. Backtest skipped.")


if __name__ == "__main__":
    main()
