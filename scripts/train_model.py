# ============================================================
# File: scripts/train_model.py
# Purpose: Train NBA predictive models (team-level & player-level)
# Author: <your name / org>
# Last Updated: 2025-02-21
#
# Notes:
# - Integrated with 3-workflow CI/CD pipeline
# - Models stored in /models, metrics in /results
# - Improved modularity, logging, preprocessing consistency
# - Unified artifact naming + versioning for CI
# ============================================================

import argparse
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss,
    mean_squared_error
)
from sklearn.linear import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier, XGBRegressor

from core.paths import (
    TRAINING_FEATURES_FILE,
    DATA_DIR,
    RESULTS_DIR,
    ARCHIVE_DIR,
    MODELS_DIR,
    ensure_dirs,
)
from core.log_config import init_global_logger
from core.exceptions import DataError, FileError

logger = init_global_logger()

PLAYER_FEATURES_FILE = DATA_DIR / "player_features.csv"


# ============================================================
# Utility Functions
# ============================================================

def archive(path: Path, prefix: str):
    """Archive old model versions with timestamp."""
    if path.exists():
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        archive_path = ARCHIVE_DIR / f"{prefix}_{ts}{''.join(path.suffixes)}"
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(path.read_bytes())
        logger.info(f"📦 Archived old artifact → {archive_path}")


def safe_cv(n_samples: int, max_cv: int = 5) -> int:
    return max(2, min(max_cv, n_samples)) if n_samples >= 2 else 2


# ============================================================
# Preprocessing
# ============================================================

def preprocess_features(df: pd.DataFrame, model_type: str):
    """Return cleaned feature matrix X and target y."""
    drop_cols = {
        "game_id", "home_team", "away_team",
        "label", "margin", "overtime", "outcome_category"
    }

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    X = X.fillna(0)

    # XGBoost prefers numeric-dense matrices
    if model_type == "xgb":
        X = pd.get_dummies(X, drop_first=True).fillna(0)

    return X


# ============================================================
# Team Model Training
# ============================================================

def train_team_model(target: str = "label", model_type: str = "logistic", tune: bool = False):
    ensure_dirs()

    if not TRAINING_FEATURES_FILE.exists():
        raise DataError(f"Missing training features: {TRAINING_FEATURES_FILE}")

    df = pd.read_csv(TRAINING_FEATURES_FILE)

    if target not in df.columns:
        raise DataError(f"Missing target column '{target}'")

    y = df[target].copy()

    # Encode label
    if target == "label":
        mapping = {"HOME": 1, "AWAY": 0, "home": 1, "away": 0}
        y = y.map(mapping)
        if y.isna().any():
            raise DataError("Invalid classes in label target")

    if df.shape[0] < 3:
        raise DataError("Not enough samples to train")

    X = preprocess_features(df, model_type=model_type)

    # Train/test split
    stratify = y if target == "label" and y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    numeric = X.select_dtypes(include="number").columns
    categorical = X.select_dtypes(exclude="number").columns

    # =======================================================
    # Model Selection
    # =======================================================

    if target == "margin":
        model = LinearRegression()
        model.fit(X_train, y_train)

    elif target == "outcome_category":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

    else:
        if model_type == "logistic":
            pre = ColumnTransformer([
                ("num", StandardScaler(), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
            ])
            model = Pipeline([
                ("pre", pre),
                ("clf", LogisticRegression(max_iter=1000))
            ])
            model.fit(X_train, y_train)

        elif model_type == "rf":
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)

        elif model_type == "xgb":
            xgb = XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
            )

            if tune and len(X_train) > 20:
                search = RandomizedSearchCV(
                    xgb,
                    param_distributions={
                        "n_estimators": [200, 400, 800],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "max_depth": [3, 5, 7],
                        "subsample": [0.6, 0.8, 1.0],
                        "colsample_bytree": [0.6, 0.8, 1.0],
                    },
                    n_iter=12,
                    scoring="roc_auc",
                    cv=safe_cv(len(X_train), max_cv=3),
                    random_state=42,
                    n_jobs=-1,
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
            else:
                model = xgb
                model.fit(X_train, y_train)

        else:
            raise DataError(f"Unknown model type '{model_type}'")

    # ============================================================
    # Metrics
    # ============================================================

    metrics = {}

    if target == "label":
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred
        )

        metrics.update({
            "accuracy": accuracy_score(y_test, y_pred),
            "log_loss": log_loss(y_test, y_prob),
            "brier": brier_score_loss(y_test, y_prob),
            "auc": roc_auc_score(y_test, y_prob),
        })

        cv_folds = safe_cv(len(X))
        cv_scores = cross_val_score(model, X, y, cv=cv_folds)
        metrics.update({
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        })

    elif target == "margin":
        y_pred = model.predict(X_test)
        metrics["rmse"] = mean_squared_error(y_test, y_pred, squared=False)

    else:
        y_pred = model.predict(X_test)
        metrics["accuracy"] = accuracy_score(y_test, y_pred)

    # ============================================================
    # Save Model + Metrics
    # ============================================================

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_name = f"game_predictor_{model_type}.pkl"
    model_path = MODELS_DIR / model_name

    archive(model_path, prefix="team_model")

    joblib.dump(
        {"model": model, "features": list(X.columns), "target": target},
        model_path
    )
    logger.info(f"💾 Saved model → {model_path}")

    metrics_df = pd.DataFrame([metrics])
    metrics_file = RESULTS_DIR / f"metrics_team_{target}_{model_type}.csv"
    metrics_df.to_csv(metrics_file, index=False)

    return metrics


# ============================================================
# Player Model Training
# ============================================================

def train_player_model(target="pts"):
    ensure_dirs()

    if not PLAYER_FEATURES_FILE.exists():
        raise DataError("Missing player feature file")

    df = pd.read_csv(PLAYER_FEATURES_FILE)

    required = {"player_avg_pts", "player_avg_ast", "player_avg_reb"}
    if not required.issubset(df.columns):
        raise DataError(f"Missing required player features {required}")

    X = df[list(required)].fillna(0)
    y = df[target].fillna(0)

    if len(X) < 3:
        raise DataError("Not enough samples for player model")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    metrics = {"rmse": rmse}

    # Save player model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"player_predictor_{target}.pkl"
    archive(model_path, prefix="player_model")

    joblib.dump({"model": model, "features": list(X.columns)}, model_path)
    logger.info(f"💾 Saved player model → {model_path}")

    return metrics


# ============================================================
# Main Entrypoint
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="label")
    parser.add_argument("--model_type", default="logistic")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--player", action="store_true")
    parser.add_argument("--player_target", default="pts")

    args = parser.parse_args()

    if args.player:
        return train_player_model(target=args.player_target)

    return train_team_model(
        target=args.target,
        model_type=args.model_type,
        tune=args.tune,
    )


if __name__ == "__main__":
    main()
