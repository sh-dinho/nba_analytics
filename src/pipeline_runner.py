# ============================================================
# File: src/pipeline_runner.py
# Purpose: Unified orchestration of NBA analytics pipeline (v2.9)
# Version: 2.9
# Author: Your Team
# Date: December 2025
# ============================================================

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss
from scipy.stats import ks_2samp

from src.config.config_loader import Config
from src.model.train_model import train_model, predict_outcomes
from src.model.predict import load_model, predict_schedule
from src.features.feature_engineering import prepare_features
from src.utils.common import (
    configure_logging,
    save_dataframe,
    load_dataframe,
    clean_data,
)
from src.schedule.contract import validate_team_schedule
from src.schedule.pipeline_historical import (
    add_features,
    add_predicted_win,
    add_team_strength as historical_add_team_strength,
    generate_daily_rankings,
    merge_rankings_history,
    save_rankings,
)
from src.schemas.normalize import normalize
from src.ranking.bet import generate_betting_recommendations

# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------

DATA_DIR = Path("data")
HISTORY_DIR = DATA_DIR / "history"
CACHE_DIR = DATA_DIR / "cache"
ARCHIVE_DIR = DATA_DIR / "archive"

MASTER_FILE = HISTORY_DIR / "historical_schedule.parquet"
ENRICHED_FILE = CACHE_DIR / "master_schedule.parquet"

CONFIG_PATH = "src/config/pipeline_config.yaml"

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------


def setup():
    cfg = Config(CONFIG_PATH)
    logger = configure_logging(
        name="pipeline_runner",
        level=cfg.logging.get("level", "INFO"),
    )
    for p in [HISTORY_DIR, CACHE_DIR, ARCHIVE_DIR, Path("logs/interpretability")]:
        p.mkdir(parents=True, exist_ok=True)
    return logger, cfg


# ---------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------


def ingest_historical() -> pd.DataFrame:
    if not MASTER_FILE.exists():
        raise FileNotFoundError(f"{MASTER_FILE} not found.")
    raw_df = load_dataframe(MASTER_FILE)
    if raw_df.empty:
        raise FileNotFoundError(f"{MASTER_FILE} is empty.")
    df = clean_data(raw_df)
    df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce")
    df = df.dropna(subset=["startDate"]).drop_duplicates(subset=["gameId"])
    df = df.sort_values("startDate").reset_index(drop=True)
    save_dataframe(df, MASTER_FILE)
    return df


# ---------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------


def enrich_schedule(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = add_features(df)
    df = historical_add_team_strength(
        df,
        streak_weight=cfg.strength.get("streak_weight", 0.3),
        diff_weight=cfg.strength.get("diff_weight", 0.4),
        winpct_weight=cfg.strength.get("winpct_weight", 0.3),
    )
    df = add_predicted_win(df)
    df = normalize(df, "enriched_schedule")
    validate_team_schedule(df)
    save_dataframe(df, ENRICHED_FILE)
    return df


# ---------------------------------------------------------------------
# Rankings & Betting
# ---------------------------------------------------------------------


def generate_and_archive_rankings(df: pd.DataFrame) -> pd.DataFrame:
    rankings = generate_daily_rankings(df)
    save_rankings(rankings)
    _ = merge_rankings_history()
    return rankings


def recommend_bets(enriched_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    class BettingCfg:
        threshold = cfg.betting.get("threshold", 0.6)

    return generate_betting_recommendations(enriched_df, BettingCfg())


# ---------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------


def monitor_model(
    df: pd.DataFrame,
    features: list,
    brier_threshold: float = 0.25,
    drift_threshold: float = 0.2,
) -> dict:
    results = {"retrain_needed": False}

    if "homeWin" in df.columns and "predicted_prob" in df.columns:
        brier = brier_score_loss(df["homeWin"], df["predicted_prob"])
        results["brier_score"] = brier
        if brier > brier_threshold:
            logging.warning(
                f"Brier score {brier:.3f} exceeds threshold {brier_threshold}"
            )
            results["retrain_needed"] = True
    else:
        results["brier_score"] = None

    if "startDate" in df.columns:
        df = df.sort_values("startDate")

    recent, past = df.tail(30), df.tail(60).head(30)

    drift_scores = {}
    for f in features:
        if f in df.columns:
            try:
                stat = ks_2samp(recent[f], past[f]).statistic
                drift_scores[f] = stat
                if stat > drift_threshold:
                    logging.warning(f"Drift alert: {f} KS={stat:.3f}")
                    results["retrain_needed"] = True
            except Exception:
                drift_scores[f] = None
    results["feature_drift"] = drift_scores
    return results


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------


def run_pipeline() -> None:
    logger, cfg = setup()
    logger.info("===== PIPELINE START =====")
    try:
        # 1. Ingest historical schedule
        hist_df = ingest_historical()

        # 2. Enrich schedule with derived stats
        enriched_df = enrich_schedule(hist_df, cfg)

        # 3. Prepare engineered features for training
        train_features = prepare_features(enriched_df)
        model_cfg = cfg.model

        # 4. Train model
        model, model_path = train_model(train_features, model_cfg)

        # 5. Predict on engineered features
        preds = predict_outcomes(model, train_features.copy(), model_cfg)

        # 6. Merge predictions back into enriched schedule (preserves homeTeam/awayTeam)
        enriched_df = enriched_df.merge(
            preds[["startDate", "homeWin", "predicted_outcome", "predicted_prob"]],
            on=["startDate", "homeWin"],
            how="left",
        )
        save_dataframe(enriched_df, ENRICHED_FILE)

        # 7. Monitor model performance
        metrics = monitor_model(enriched_df, model_cfg["features"])
        save_path = CACHE_DIR / f"monitoring_{datetime.now().strftime('%Y-%m-%d')}.json"
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # 8. Retraining if drift detected
        if bool(metrics.get("retrain_needed", False)):
            logger.warning("Retraining triggered...")
            model, model_path = train_model(train_features, model_cfg)
            preds = predict_outcomes(model, train_features.copy(), model_cfg)
            enriched_df = enriched_df.merge(
                preds[["startDate", "homeWin", "predicted_outcome", "predicted_prob"]],
                on=["startDate", "homeWin"],
                how="left",
            )
            save_dataframe(enriched_df, ENRICHED_FILE)

        # 9. Generate rankings and betting recommendations
        rankings = generate_and_archive_rankings(enriched_df)
        recs = recommend_bets(enriched_df, cfg)
        save_dataframe(recs, CACHE_DIR / "bets.parquet")

        # 10. Save latest predictions (merge engineered predictions back into enriched schedule)
        latest_model = load_model(model_path)
        if latest_model:
            # Run predictions on engineered features
            fresh_preds = predict_schedule(
                latest_model, train_features.copy(), model_cfg
            )

            # Merge predictions back into enriched_df to keep identifiers
            fresh_predictions = enriched_df.merge(
                fresh_preds[
                    ["startDate", "homeWin", "predicted_outcome", "predicted_prob"]
                ],
                on=["startDate", "homeWin"],
                how="left",
            )

            # Clean up duplicate column names (remove _x/_y suffixes)
            for col in ["predicted_outcome", "predicted_prob"]:
                if f"{col}_y" in fresh_predictions.columns:
                    fresh_predictions[col] = fresh_predictions[f"{col}_y"]
                    # Drop all suffixed variants except the canonical one
                    drop_cols = [
                        c
                        for c in fresh_predictions.columns
                        if c.startswith(col + "_") and c != col
                    ]
                    fresh_predictions.drop(columns=drop_cols, inplace=True)

            save_dataframe(fresh_predictions, CACHE_DIR / "latest_predictions.parquet")
            fresh_predictions.to_csv(CACHE_DIR / "latest_predictions.csv", index=False)

        # 11. Enhanced logging
        logger.info("Pipeline completed successfully.")
        logger.info(f"Rankings saved to {(CACHE_DIR / 'rankings.parquet').resolve()}")
        logger.info(
            f"Betting recommendations saved to {(CACHE_DIR / 'bets.parquet').resolve()}"
        )
        logger.info(
            f"Latest predictions saved to {(CACHE_DIR / 'latest_predictions.parquet').resolve()}"
        )
        logger.info("===== PIPELINE END (SUCCESS) =====")

    except Exception as e:  # noqa: BLE001
        logger.exception(f"Pipeline failed: {e}")
        logger.info("===== PIPELINE END (FAILED) =====")
        raise


if __name__ == "__main__":
    run_pipeline()
