# File: scripts/train_and_backtest.py
import argparse
import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from nba_analytics_core.data import (
    fetch_historical_games,
    build_team_stats,
    build_matchup_features
)
from nba_analytics_core.odds import fetch_odds
from scripts.simulate_bankroll import simulate_bankroll
from nba_analytics_core.notifications import send_telegram_message

DEFAULT_MODEL_PATH = "models/classification_model.pkl"
RESULTS_DIR = "results"

# ----------------------------
# Logging configuration
# ----------------------------
logger = logging.getLogger("train_backtest")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)


# ----------------------------
# Utility helpers
# ----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamp_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def add_timestamp_to_path(path: str) -> str:
    """
    Insert a timestamp before file extension.
    e.g. results/backtest_summary.csv -> results/backtest_summary_2025-12-01_18-20-00.csv
    """
    base, ext = os.path.splitext(path)
    return f"{base}_{timestamp_str()}{ext}"


def is_valid_feature_vector(vec: Any) -> bool:
    try:
        arr = np.array(vec, dtype=float)
        if arr.ndim != 1:
            return False
        if not np.isfinite(arr).all():
            return False
        return True
    except Exception:
        return False


def safe_fetch_odds(home_team: str, away_team: str) -> Optional[Dict[str, Any]]:
    try:
        odds = fetch_odds(home_team=home_team, away_team=away_team)
        if not odds:
            return None
        home_odds = odds.get("home_odds")
        if home_odds is None or home_odds <= 1:
            # Invalid decimal odds
            return None
        return {"home_odds": float(home_odds), **{k: v for k, v in odds.items() if k != "home_odds"}}
    except Exception as e:
        logger.warning(f"Failed to fetch odds for {home_team} vs {away_team}: {e}")
        return None


# ----------------------------
# Model training
# ----------------------------
def train_model(seasons: List[str], model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    logger.info(f"Starting training on seasons: {', '.join(seasons)}")

    X, y = [], []
    feature_lengths = set()
    skipped_games = 0

    for season in seasons:
        games = fetch_historical_games(season)
        team_stats = build_team_stats(games)

        for g in games:
            if not g.get("away_team") or not g.get("home_team"):
                skipped_games += 1
                continue

            feats = build_matchup_features(g["home_team"], g["away_team"], team_stats)

            if not is_valid_feature_vector(feats):
                skipped_games += 1
                continue

            feature_lengths.add(len(feats))
            X.append(feats)
            y.append(1 if g.get("home_win") else 0)

    if not X:
        raise ValueError("No valid training data collected. Check data sources and feature generation.")

    if len(feature_lengths) != 1:
        raise ValueError(f"Inconsistent feature vector lengths detected: {feature_lengths}")

    # Build pipeline with scaling + logistic regression
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X, y)
    ensure_dir(os.path.dirname(model_path) or ".")

    metadata = {
        "trained_on_seasons": seasons,
        "model_type": "LogisticRegression",
        "pipeline_steps": ["StandardScaler", "LogisticRegression"],
        "feature_length": list(feature_lengths)[0],
        "skipped_games_during_training": skipped_games,
        "timestamp": datetime.now().isoformat()
    }

    joblib.dump({"model": pipeline, "metadata": metadata}, model_path)
    logger.info(f"Model trained and saved to {model_path}")
    logger.info(f"Skipped games during training due to missing/invalid features: {skipped_games}")

    return {"model": pipeline, "metadata": metadata}


# ----------------------------
# Backtesting
# ----------------------------
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

    logger.info(f"Starting backtest on seasons: {', '.join(seasons)} with strategy={strategy}, max_fraction={max_fraction}")

    # Load model and metadata
    bundle = joblib.load(model_path)
    model = bundle["model"]
    metadata = bundle.get("metadata", {})

    summary_results: List[Dict[str, Any]] = []
    detailed_results: List[Dict[str, Any]] = []

    ensure_dir(RESULTS_DIR)
    ensure_dir(export_bankroll_progression_dir)

    skipped_games_total = 0
    skipped_odds_total = 0

    for season in seasons:
        games = fetch_historical_games(season)
        team_stats = build_team_stats(games)
        bets: List[Dict[str, Any]] = []
        skipped_games_season = 0
        skipped_odds_season = 0

        for g in games:
            home_team = g.get("home_team")
            away_team = g.get("away_team")
            home_win = g.get("home_win")
            game_id = g.get("game_id")

            if not home_team or not away_team:
                skipped_games_season += 1
                continue

            feats = build_matchup_features(home_team, away_team, team_stats)
            if not is_valid_feature_vector(feats) or len(feats) != metadata.get("feature_length", len(feats)):
                skipped_games_season += 1
                continue

            # Predict home win probability
            prob = float(model.predict_proba([feats])[0][1])

            # Fetch odds safely
            odds = safe_fetch_odds(home_team=home_team, away_team=away_team)
            if not odds:
                skipped_odds_season += 1
                continue

            home_odds = odds["home_odds"]
            implied_prob = 1.0 / home_odds
            edge = prob - implied_prob
            ev = (prob * (home_odds - 1.0)) - (1.0 - prob)

            bets.append({
                "season": season,
                "game_id": game_id,
                "home_team": home_team,
                "away_team": away_team,
                "decimal_odds": home_odds,
                "prob": prob,
                "implied_prob": implied_prob,
                "edge": edge,
                "ev": ev,
                "home_win": home_win
            })

        skipped_games_total += skipped_games_season
        skipped_odds_total += skipped_odds_season

        if not bets:
            logger.info(f"No bets generated for season {season}. Skipped games={skipped_games_season}, skipped odds={skipped_odds_season}")
            continue

        df_bets = pd.DataFrame(bets)

        # Simulate bankroll and capture progression per season
        progression_df, metrics = simulate_bankroll(df_bets, strategy=strategy, max_fraction=max_fraction)

        # Save bankroll progression for this season
        season_progression_path = os.path.join(
            export_bankroll_progression_dir,
            f"bankroll_progression_{season}_{timestamp_str()}.csv"
        )
        progression_df.to_csv(season_progression_path, index=False)
        logger.info(f"Bankroll progression for {season} exported to {season_progression_path}")

        # Compute per-season predictive metrics
        acc = accuracy_score(df_bets["home_win"], (df_bets["prob"] >= 0.5).astype(int))
        ll = log_loss(df_bets["home_win"], df_bets["prob"])
        brier = brier_score_loss(df_bets["home_win"], df_bets["prob"])

        summary_results.append({
            "season": season,
            "roi": metrics.get("roi"),
            "win_rate": metrics.get("win_rate"),
            "final_bankroll": metrics.get("final_bankroll"),
            "num_bets": len(df_bets),
            "accuracy": acc,
            "log_loss": ll,
            "brier": brier,
            "skipped_games": skipped_games_season,
            "skipped_odds": skipped_odds_season
        })
        detailed_results.extend(bets)

    if not summary_results:
        logger.warning("Backtest produced no results. Check data sources, odds availability, and feature generation.")
        return pd.DataFrame(), pd.DataFrame()

    # Export results with timestamped filenames
    ensure_dir(RESULTS_DIR)
    export_summary_ts = add_timestamp_to_path(export_summary)
    export_detailed_ts = add_timestamp_to_path(export_detailed)

    df_summary = pd.DataFrame(summary_results)
    df_detailed = pd.DataFrame(detailed_results)

    df_summary.to_csv(export_summary_ts, index=False)
    df_detailed.to_csv(export_detailed_ts, index=False)

    logger.info(f"Backtest summary exported to {export_summary_ts}")
    logger.info(f"Detailed bets exported to {export_detailed_ts}")
    logger.info(f"Total skipped games: {skipped_games_total}, total skipped odds: {skipped_odds_total}")

    # Telegram notification enrichment
    if notify:
        avg_roi = df_summary["roi"].mean()
        avg_acc = df_summary["accuracy"].mean()
        total_bets = df_summary["num_bets"].sum()
        total_final_bankroll = df_summary["final_bankroll"].iloc[-1] if not df_summary.empty else None

        best_season = df_summary.loc[df_summary["roi"].idxmax()]
        worst_season = df_summary.loc[df_summary["roi"].idxmin()]

        # Optional simple Sharpe-like ratio using ROI std (risk-adjusted, illustrative)
        roi_std = df_summary["roi"].std()
        sharpe_like = (avg_roi / roi_std) if roi_std and roi_std != 0 else None

        msg_lines = [
            "📊 Backtest Summary",
            f"Seasons: {', '.join(seasons)}",
            f"Total bets: {total_bets}",
            f"Avg ROI: {avg_roi:.4f}",
            f"Avg Accuracy: {avg_acc:.4f}",
            f"Best season: {best_season['season']} (ROI {best_season['roi']:.4f})",
            f"Worst season: {worst_season['season']} (ROI {worst_season['roi']:.4f})"
        ]

        if total_final_bankroll is not None:
            msg_lines.append(f"Final bankroll (last season): {total_final_bankroll:.2f}")
        if sharpe_like is not None:
            msg_lines.append(f"Sharpe-like: {sharpe_like:.3f}")

        msg = "\n".join(msg_lines)
        try:
            send_telegram_message(msg)
            logger.info("Telegram notification sent.")
        except Exception as e:
            logger.warning(f"Failed to send Telegram notification: {e}")

    return df_summary, df_detailed


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and backtest NBA model")

    # Core params
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2021-22", "2022-23", "2023-24"],
        help="Seasons to use (e.g., 2021-22 2022-23 2023-24)"
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtesting after training"
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Train only, skip backtesting"
    )

    # Model IO
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to save/load the model bundle (pipeline + metadata)"
    )

    # Backtest outputs
    parser.add_argument(
        "--export_summary",
        type=str,
        default=os.path.join(RESULTS_DIR, "backtest_summary.csv"),
        help="Path to export backtest summary (timestamp appended automatically)"
    )
    parser.add_argument(
        "--export_detailed",
        type=str,
        default=os.path.join(RESULTS_DIR, "backtest_detailed.csv"),
        help="Path to export detailed bets (timestamp appended automatically)"
    )
    parser.add_argument(
        "--bankroll_progression_dir",
        type=str,
        default=os.path.join(RESULTS_DIR, "bankroll_progression"),
        help="Directory to export per-season bankroll progression CSVs"
    )

    # Betting strategy
    parser.add_argument(
        "--strategy",
        choices=["kelly", "flat"],
        default="kelly",
        help="Bet sizing strategy"
    )
    parser.add_argument(
        "--max_fraction",
        type=float,
        default=0.05,
        help="Max fraction of bankroll to bet (used in kelly or as flat stake fraction)"
    )

    # Notifications
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send Telegram notification after backtest"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Train
    bundle = train_model(args.seasons, model_path=args.model_path)

    # Backtest condition
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