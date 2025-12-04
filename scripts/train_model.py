# ============================================================
# File: scripts/train_model.py
# Purpose: Train predictive models on NBA features (team + player)
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
from sklearn.linear_model import LogisticRegression, LinearRegression
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
    LOGS_DIR,
    ensure_dirs,
)
from core.log_config import init_global_logger
from core.exceptions import DataError, FileError

logger = init_global_logger()

# Model artifact path (team model)
MODEL_FILE_PKL = LOGS_DIR / "team_model.pkl"
# Player features file path
PLAYER_FEATURES_FILE = DATA_DIR / "player_features.csv"


def _archive_artifact(path: Path, prefix: str):
    """Archive an existing artifact before overwrite."""
    if path.exists():
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        archive_path = ARCHIVE_DIR / f"{prefix}_{ts}{''.join(path.suffixes)}"
        try:
            archive_path.write_bytes(path.read_bytes())
            logger.info(f"📦 Archived artifact {path.name} → {archive_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to archive {path}: {e}")


def _safe_cv(n_samples: int, max_cv: int = 5) -> int:
    """Return a safe cross-validation folds count based on sample size."""
    return max(2, min(max_cv, n_samples)) if n_samples >= 2 else 2


def train_team_model(target: str = "label", model_type: str = "logistic", tune: bool = False):
    """Train a team-level model on training features."""
    ensure_dirs(strict=False)

    if not TRAINING_FEATURES_FILE.exists():
        raise DataError(f"Training features file not found: {TRAINING_FEATURES_FILE}")

    try:
        df = pd.read_csv(TRAINING_FEATURES_FILE)
    except Exception as e:
        raise FileError(
            f"Failed to read training features {TRAINING_FEATURES_FILE}",
            file_path=str(TRAINING_FEATURES_FILE),
        ) from e

    if target not in df.columns:
        raise DataError(f"Training data missing target column '{target}'")

    # For classification targets, ensure at least two classes
    if target in ("label", "outcome_category") and df[target].nunique() < 2:
        raise DataError(f"Target '{target}' has only one class. Cannot train model.")

    # Features: drop identifiers and target columns
    drop_cols = {"game_id", "home_team", "away_team", "label", "margin", "overtime", "outcome_category"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df[target].copy()

    # Encode target if necessary
    if target == "label" and y.dtype == "object":
        mapping = {"HOME": 1, "AWAY": 0, "home": 1, "away": 0}
        y = y.map(mapping)
        if y.isna().any():
            raise DataError("Target 'label' contains unknown classes; expected HOME/AWAY.")

    # Ensure numeric features for XGBoost
    if model_type.lower() == "xgb":
        X = pd.get_dummies(X, drop_first=True).fillna(0)
    else:
        X = X.fillna(0)

    if X.shape[1] == 0:
        raise DataError("No usable features after preprocessing.")

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    if len(X) < 3:
        raise DataError("Not enough samples to train (need at least 3 rows).")

    # Stratify for classification targets
    stratify = y if target in ("label", "outcome_category") and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # Select and train model
    model_type = model_type.lower()
    if target == "margin":
        model = LinearRegression()
        model.fit(X_train, y_train)

    elif target == "outcome_category":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

    else:  # binary label
        if model_type == "rf":
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)

        elif model_type == "xgb":
            base_model = XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
            )
            if tune and len(X_train) > 20:
                param_dist = {
                    "n_estimators": [200, 400, 800],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0],
                }
                cv_folds = _safe_cv(len(X_train), max_cv=3)
                logger.info(f"Using cv={cv_folds} for hyperparameter search (n_samples={len(X_train)})")
                search = RandomizedSearchCV(
                    base_model,
                    param_distributions=param_dist,
                    n_iter=12,
                    scoring="roc_auc",
                    cv=cv_folds,
                    verbose=1,
                    random_state=42,
                    n_jobs=-1,
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                logger.info(f"✅ Best XGBoost params: {search.best_params_}")
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    early_stopping_rounds=50,
                    verbose=False,
                )
            else:
                base_model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    early_stopping_rounds=50,
                    verbose=False,
                )
                model = base_model

        else:  # logistic (default), handle_unknown for categoricals
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numeric_features),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                ]
            )
            model = Pipeline([
                ("preprocessor", preprocessor),
                ("logreg", LogisticRegression(max_iter=1000)),
            ])
            model.fit(X_train, y_train)

    # Evaluate
    metrics: dict = {}
    if target == "label":
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            df_vals = model.decision_function(X_test)
            y_prob = (df_vals - df_vals.min()) / (df_vals.max() - df_vals.min() + 1e-9)
        else:
            y_prob = pd.Series(y_pred).astype(float)

        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        metrics["log_loss"] = log_loss(y_test, y_prob)
        metrics["brier"] = brier_score_loss(y_test, y_prob)
        metrics["auc"] = roc_auc_score(y_test, y_prob)

        cv_folds = _safe_cv(len(X))
        cv_scores = cross_val_score(model, X, y, cv=cv_folds)
        metrics["cv_accuracy_mean"] = cv_scores.mean()
        metrics["cv_accuracy_std"] = cv_scores.std()

    elif target == "margin":
        y_pred = model.predict(X_test)
        metrics["rmse"] = mean_squared_error(y_test, y_pred, squared=False)
        cv_folds = _safe_cv(len(X))
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="neg_root_mean_squared_error")
        metrics["cv_rmse_mean"] = -cv_scores.mean()
        metrics["cv_rmse_std"] = cv_scores.std()

    elif target == "outcome_category":
        y_pred = model.predict(X_test)
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        cv_folds = _safe_cv(len(X))
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")
        metrics["cv_accuracy_mean"] = cv_scores.mean()
        metrics["cv_accuracy_std"] = cv_scores.std()

    # Save model artifact with archiving
    try:
        _archive_artifact(MODEL_FILE_PKL, prefix="team_model")
        MODEL_FILE_PKL.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "features": list(X.columns), "target": target}, MODEL_FILE_PKL)
        logger.info(f"✅ Team model trained on target='{target}' and saved to {MODEL_FILE_PKL}")
    except Exception as e:
        raise FileError(f"Failed to save model artifact {MODEL_FILE_PKL}", file_path=str(MODEL_FILE_PKL)) from e

    # Save metrics
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_df = pd.DataFrame([metrics])
        metrics_file = RESULTS_DIR / f"metrics_team_{target}_{model_type}.csv"
        metrics_df.to_csv(metrics_file, index=False)
        logger.info(f"📊 Team metrics saved to {metrics_file}: {metrics}")
    except Exception as e:
        raise FileError(f"Failed to write metrics file {RESULTS_DIR}", file_path=str(RESULTS_DIR)) from e


def train_player_model(target: str = "pts"):
    """Train a simple player-level regression model."""
    ensure_dirs(strict=False)

    if not PLAYER_FEATURES_FILE.exists():
        raise DataError(f"Player features file not found: {PLAYER_FEATURES_FILE}")

    try:
        df = pd.read_csv(PLAYER_FEATURES_FILE)
    except Exception as e:
        raise FileError(
            f"Failed to read player features {PLAYER_FEATURES_FILE}",
            file_path=str(PLAYER_FEATURES_FILE),
        ) from e

    if target not in df.columns:
        raise DataError(f"Player data missing '{target}' column")

    required_feats = {"player_avg_pts", "player_avg_ast", "player_avg_reb"}
    missing_feats = required_feats - set(df.columns)
    if missing_feats:
        raise DataError(f"Player features missing columns: {missing_feats}")

    X = df[["player_avg_pts", "player_avg_ast", "player_avg_reb"]].fillna(0)
    y = df[target].fillna(0)

    if len(X) < 3:
        raise DataError("Not enough player samples to train (need at least 3 rows).")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    logger.info(f"✅ Player model trained on target='{target}' RMSE={rmse:.3f}")

    # Save metrics
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_df = pd.DataFrame([{"rmse": rmse}])
        metrics_file = RESULTS_DIR / f"metrics_player_{target}.csv"
        metrics_df.to_csv(metrics_file, index=False)
        logger.info(f"📊 Player metrics saved to {metrics_file}")
    except Exception as e:
        raise FileError(f"Failed to write player metrics file {RESULTS_DIR}", file_path=str(RESULTS_DIR)) from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NBA prediction models")
    parser.add_argument("--target", type=str, default="label",
                        help="Team model target: label (binary), margin (regression), outcome_category (multi-class)")
    parser.add_argument("--model_type", type=str, default="logistic",
                        help="Team model type for label: logistic, rf, xgb")
    parser.add_argument("--tune", action="store_true",
                        help="Enable hyperparameter tuning for XGBoost (team model only)")
    parser.add_argument("--player", action="store_true",
                        help="Train player-level model instead of team model")
    parser.add_argument("--player_target", type=str, default="pts",
                        help="Target column for player model (default: pts)")
    args = parser.parse_args()

    if args.player:
        train_player_model(target=args.player_target)
    else:
        train_team_model(target=args.target, model_type=args.model_type, tune=args.tune)
