from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Spread Model Training (ATS + Margin)
# File: src/model/train_spread.py
# Author: Sadiq
#
# Description:
#     Trains TWO models when sportsbook data is available:
#       1. Regression → predict margin (score - opponent_score)
#       2. Classification → predict ATS cover (1 = cover)
#
#     When sportsbook spread data is NOT available:
#       - Trains ONLY the regression model (margin)
#       - Skips ATS classification cleanly
#
#     Uses:
#       - Canonical long snapshot
#       - FeatureBuilder v4
#       - training_core helpers (numeric-only + NaN handling)
#       - Model Registry v4
# ============================================================

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    log_loss,
)
from sklearn.model_selection import train_test_split

from src.config.paths import LONG_SNAPSHOT, MODEL_DIR
from src.features.builder import FeatureBuilder
from src.model.registry import register_model
from src.model.training_core import _drop_non_numeric_features, _fill_missing_values


# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------


@dataclass
class SpreadTrainingConfig:
    feature_version: str = "v4"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    max_depth: Optional[int] = None
    min_samples_leaf: int = 2


# ------------------------------------------------------------
# Load canonical long snapshot
# ------------------------------------------------------------


def _load_long() -> pd.DataFrame:
    df = pd.read_parquet(LONG_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"])  # keep datetime64[ns] for merging
    return df


# ------------------------------------------------------------
# Build features + targets
# ------------------------------------------------------------


def _build_training_frame(cfg: SpreadTrainingConfig) -> Tuple[pd.DataFrame, bool]:
    """
    Build training frame with features + targets.

    Returns:
        merged: DataFrame with features + margin + ats_cover (may be None)
        do_classification: bool indicating whether ATS classification is possible
    """
    df = _load_long()

    fb = FeatureBuilder(version=cfg.feature_version)
    features = fb.build_from_long(df)

    # Only home rows → one row per game
    df_home = df[df["is_home"] == 1].copy()

    # Regression target
    df_home["margin"] = df_home["score"] - df_home["opponent_score"]

    # ATS classification only if sportsbook data exists
    if "spread_line" not in df_home.columns:
        logger.warning(
            "[Spread] No spread_line column found — ATS classification disabled."
        )
        df_home["ats_cover"] = None
        do_classification = False
    else:
        df_home["ats_cover"] = (df_home["margin"] > df_home["spread_line"]).astype(int)
        do_classification = True

    merged = features.merge(
        df_home[["game_id", "team", "date", "margin", "ats_cover"]],
        on=["game_id", "team", "date"],
        how="inner",
    )

    # Need margin for regression; ats_cover may be None if classification disabled
    merged = merged.dropna(subset=["margin"])
    return merged, do_classification


# ------------------------------------------------------------
# Time-based split
# ------------------------------------------------------------


def _time_split(df: pd.DataFrame, cfg: SpreadTrainingConfig):
    df = df.sort_values("date")
    unique_dates = sorted(df["date"].unique())

    cutoff_idx = int(len(unique_dates) * (1 - cfg.test_size))
    cutoff_date = unique_dates[cutoff_idx]

    train_df = df[df["date"] < cutoff_date]
    test_df = df[df["date"] >= cutoff_date]

    feature_cols = [
        c
        for c in df.columns
        if c not in ("margin", "ats_cover", "date", "game_id", "team")
    ]

    return (
        train_df[feature_cols],
        test_df[feature_cols],
        train_df["margin"],
        test_df["margin"],
        train_df["ats_cover"],
        test_df["ats_cover"],
        feature_cols,
        df["date"].min(),
        df["date"].max(),
    )


# ------------------------------------------------------------
# Save model + metadata
# ------------------------------------------------------------


def _save_model(model, model_type: str, feature_cols: List[str], metrics: dict):
    version = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    out_dir = MODEL_DIR / model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{version}.pkl"
    pd.to_pickle(model, model_path)

    meta = {
        "model_type": model_type,
        "version": version,
        "created_at": datetime.utcnow().isoformat(),
        "is_production": False,
        "feature_version": "v4",
        "feature_cols": feature_cols,
        "metrics": metrics,
    }

    register_model(meta)
    logger.success(f"[Spread] Saved {model_type} model → {model_path}")
    logger.success(f"[Spread] Registered {model_type} model → version={version}")


# ------------------------------------------------------------
# Main training entry point
# ------------------------------------------------------------


def train_spread_models(cfg: Optional[SpreadTrainingConfig] = None):
    cfg = cfg or SpreadTrainingConfig()
    logger.info(
        "🏀 Training spread models "
        "(regression always, ATS classification only if sportsbook lines exist)"
    )

    df, do_classification = _build_training_frame(cfg)

    (
        X_train,
        X_test,
        y_margin_train,
        y_margin_test,
        y_ats_train,
        y_ats_test,
        feature_cols,
        start_date,
        end_date,
    ) = _time_split(df, cfg)

    # Clean features
    X_train = _drop_non_numeric_features(X_train)
    X_test = _drop_non_numeric_features(X_test)
    X_train = _fill_missing_values(X_train)
    X_test = _fill_missing_values(X_test)

    # --------------------------------------------------------
    # Regression model (margin)
    # --------------------------------------------------------
    reg = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    reg.fit(X_train, y_margin_train)

    reg_train_pred = reg.predict(X_train)
    reg_test_pred = reg.predict(X_test)

    reg_metrics = {
        "target": "margin",
        "train_mae": float(mean_absolute_error(y_margin_train, reg_train_pred)),
        "test_mae": float(mean_absolute_error(y_margin_test, reg_test_pred)),
        "train_rmse": float(
            mean_squared_error(y_margin_train, reg_train_pred, squared=False)
        ),
        "test_rmse": float(
            mean_squared_error(y_margin_test, reg_test_pred, squared=False)
        ),
        "train_start_date": str(start_date),
        "train_end_date": str(end_date),
    }

    _save_model(reg, "spread_regression", feature_cols, reg_metrics)

    # --------------------------------------------------------
    # Classification model (ATS cover) — optional
    # --------------------------------------------------------
    if do_classification and y_ats_train.notna().any():
        logger.info(
            "[Spread] Training ATS classification model (spread_line detected)."
        )

        # Filter out any rows where ats_cover is NaN (defensive)
        mask_train = y_ats_train.notna()
        mask_test = y_ats_test.notna()

        X_train_clf = X_train[mask_train]
        X_test_clf = X_test[mask_test]
        y_ats_train_clf = y_ats_train[mask_train]
        y_ats_test_clf = y_ats_test[mask_test]

        clf = LogisticRegression(max_iter=500)
        clf.fit(X_train_clf, y_ats_train_clf)

        clf_train_pred = clf.predict(X_train_clf)
        clf_test_pred = clf.predict(X_test_clf)

        clf_train_prob = clf.predict_proba(X_train_clf)[:, 1]
        clf_test_prob = clf.predict_proba(X_test_clf)[:, 1]

        clf_metrics = {
            "target": "ats_cover",
            "train_accuracy": float(accuracy_score(y_ats_train_clf, clf_train_pred)),
            "test_accuracy": float(accuracy_score(y_ats_test_clf, clf_test_pred)),
            "train_log_loss": float(log_loss(y_ats_train_clf, clf_train_prob)),
            "test_log_loss": float(log_loss(y_ats_test_clf, clf_test_prob)),
            "train_start_date": str(start_date),
            "train_end_date": str(end_date),
        }

        _save_model(clf, "spread_classification", feature_cols, clf_metrics)
    else:
        logger.info(
            "[Spread] ATS classification skipped "
            "(no sportsbook spread_line available)."
        )

    logger.success(
        "🏀 Spread regression model trained "
        "(and ATS classification if sportsbook data was available)"
    )
    return reg


if __name__ == "__main__":
    train_spread_models()
