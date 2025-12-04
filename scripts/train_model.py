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

MODEL_FILE_PKL = LOGS_DIR / "team_model.pkl"
PLAYER_FEATURES_FILE = DATA_DIR / "player_features.csv"


def _archive_artifact(path: Path, prefix: str):
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
    return max(2, min(max_cv, n_samples)) if n_samples >= 2 else 2


# ============================================================
# TEAM MODEL TRAINING
# ============================================================

def train_team_model(target: str = "label", model_type: str = "logistic", tune: bool = False):
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

    if target in ("label", "outcome_category") and df[target].nunique() < 2:
        raise DataError(f"Target '{target}' has only one class. Cannot train model.")

    drop_cols = {"game_id", "home_team", "away_team", "label", "margin", "overtime", "outcome_category"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df[target].copy()

    # Encode label
    if target == "label" and y.dtype == "object":
        mapping = {"HOME": 1, "AWAY": 0, "home": 1, "away": 0}
        y = y.map(mapping)
        if y.isna().any():
            raise DataError("Target 'label' contains unknown classes; expected HOME/AWAY.")

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

    stratify = y if target in ("label", "outcome_category") and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    model_type = model_type.lower()

    # ============================================================
    # MODEL SELECTION
    # ============================================================
    if target == "margin":
        model = LinearRegression()
        model.fit(X_train, y_train)

    elif target == "outcome_category":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

    else:  # label
        # Logistic Regression
        if model_type == "logistic":
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

        # Random Forest
        elif model_type == "rf":
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)

        # XGBoost
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
            else:
                model = base_model

            # Use validation split, not test split (avoid leakage)
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=42
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False,
            )

        else:
            raise DataError(f"Unknown model_type: {model_type}")

    # ============================================================
    # METRICS
    # ============================================================

    metrics = {}

    if target == "label":
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

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

    elif target == "outcome_category":
        y_pred = model.predict(X_test)
        metrics["accuracy"] = accuracy_score(y_test, y_pred)

    # ============================================================
    # Save model and metrics
    # ============================================================

    try:
        _archive_artifact(MODEL_FILE_PKL, "team_model")
        joblib.dump({"model": model, "features": list(X.columns), "target": target}, MODEL_FILE_PKL)
    except Exception as e:
        raise FileError(f"Failed to save model artifact {MODEL_FILE_PKL}") from e

    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_df = pd.DataFrame([metrics])
        metrics_file = RESULTS_DIR / f"metrics_team_{target}_{model_type}.csv"
        metrics_df.to_csv(metrics_file, index=False)
    except Exception:
        pass

    return metrics


# ============================================================
# PLAYER MODEL TRAINING
# ============================================================

def train_player_model(target: str = "pts"):
    ensure_dirs(strict=False)

    if not PLAYER_FEATURES_FILE.exists():
        raise DataError(f"Player features file not found: {PLAYER_FEATURES_FILE}")

    df = pd.read_csv(PLAYER_FEATURES_FILE)

    if target not in df.columns:
        raise DataError(f"Player data missing '{target}' column")

    required_feats = {"player_avg_pts", "player_avg_ast", "player_avg_reb"}
    if not required_feats.issubset(df.columns):
        raise DataError("Missing required player features")

    X = df[list(required_feats)].fillna(0)
    y = df[target].fillna(0)

    if len(X) < 3:
        raise DataError("Not enough player samples to train (min=3)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)

    metrics = {"rmse": rmse}

    return metrics


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train NBA prediction models")
    parser.add_argument("--target", type=str, default="label",
                        help="Team model target: label (binary), margin (regression), outcome_category (multi-class)")
    parser.add_argument("--model_type", type=str, default="logistic",
                        help="Team model type for label: logistic, rf, xgb")
    parser.add_argument("--tune", action="store_true",
                        help="Enable hyperparameter tuning for XGBoost")
    parser.add_argument("--player", action="store_true",
                        help="Train player-level model instead of team model")
    parser.add_argument("--player_target", type=str, default="pts",
                        help="Player target column (default: pts)")
    args = parser.parse_args()

    if args.player:
        return train_player_model(target=args.player_target)

    return train_team_model(
        target=args.target,
        model_type=args.model_type,
        tune=args.tune
    )


if __name__ == "__main__":
    main()
