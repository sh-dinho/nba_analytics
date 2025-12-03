# ============================================================
# File: scripts/train_model.py
# Purpose: Train predictive models on NBA features (team + player)
# ============================================================

import argparse
import os
import pandas as pd
import joblib
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
from core.config import (
    TRAINING_FEATURES_FILE,
    MODEL_FILE_PKL,
    RESULTS_DIR,
    BASE_DATA_DIR
)
from core.log_config import setup_logger
from core.exceptions import DataError

logger = setup_logger("train_model")

PLAYER_FEATURES_FILE = os.path.join(BASE_DATA_DIR, "player_features.csv")


def safe_fit(model, X_train, y_train, X_test, y_test):
    """Try fitting with early stopping, fallback if unsupported."""
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False
        )
    except TypeError:
        logger.warning("⚠️ early_stopping_rounds not supported in this XGBoost version. Training without early stopping.")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
    return model


def _safe_cross_val(model, X, y, scoring=None):
    """Run CV only if dataset is large enough."""
    if len(X) < 30:
        logger.warning("⚠️ Not enough samples for cross-validation (n<30). Skipping CV metrics.")
        return None
    cv_scores = cross_val_score(model, X, y, cv=min(5, len(X)), scoring=scoring)
    return cv_scores


def train_team_model(target: str = "label", model_type: str = "logistic", tune: bool = False):
    if not os.path.exists(TRAINING_FEATURES_FILE):
        raise DataError(f"Training features file not found: {TRAINING_FEATURES_FILE}")

    df = pd.read_csv(TRAINING_FEATURES_FILE)
    logger.info(f"📂 Loaded team dataset with shape {df.shape}")

    if target not in df.columns:
        raise DataError(f"Training data missing '{target}' column")

    if df[target].nunique() < 2:
        raise DataError(f"Target '{target}' has only one class. Cannot train model.")

    # Features: drop identifiers and target columns
    feature_cols = [
        c for c in df.columns
        if c not in ["game_id", "home_team", "away_team",
                     "label", "margin", "overtime", "outcome_category"]
    ]
    X = df[feature_cols].fillna(0)
    y = df[target]

    logger.info(f"🔧 Using {len(feature_cols)} features for training")

    # Encode target if necessary
    if target == "label" and y.dtype == "object":
        y = y.map({"HOME": 1, "AWAY": 0})

    # For XGBoost, ensure all features are numeric
    if model_type == "xgb":
        X = pd.get_dummies(X, drop_first=True).fillna(0)

    if X.shape[1] == 0:
        raise DataError("No usable features after preprocessing.")

    # Split features into numeric and categorical (for non-XGB models)
    numeric_features = X.select_dtypes(include=["number"]).columns
    categorical_features = X.select_dtypes(exclude=["number"]).columns

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Select model
    if target == "margin":
        model = LinearRegression()
    elif target == "outcome_category":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    else:  # default binary label
        if model_type == "rf":
            model = RandomForestClassifier(n_estimators=200, random_state=42)
        elif model_type == "xgb":
            base_model = XGBClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss"
            )

            if tune and len(X_train) > 30:
                param_dist = {
                    "n_estimators": [200, 500, 1000],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0]
                }

                logger.info(f"🔍 Running hyperparameter search (n_samples={len(X_train)})")

                search = RandomizedSearchCV(
                    base_model,
                    param_distributions=param_dist,
                    n_iter=10,
                    scoring="roc_auc",
                    cv=3,
                    verbose=1,
                    random_state=42,
                    n_jobs=-1
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                logger.info(f"✅ Best XGBoost params: {search.best_params_}")

                model = safe_fit(best_model, X_train, y_train, X_test, y_test)
            else:
                if tune:
                    logger.warning("⚠️ Not enough samples for hyperparameter tuning. Skipping search.")
                model = safe_fit(base_model, X_train, y_train, X_test, y_test)
        else:
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numeric_features),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
                ]
            )
            model = Pipeline([
                ("preprocessor", preprocessor),
                ("logreg", LogisticRegression(max_iter=1000))
            ])
            model.fit(X_train, y_train)

    # Evaluate
    metrics = {}
    if target == "label":
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        metrics["log_loss"] = log_loss(y_test, y_prob)
        metrics["brier"] = brier_score_loss(y_test, y_prob)
        metrics["auc"] = roc_auc_score(y_test, y_prob)
        cv_scores = _safe_cross_val(model, X, y)
        if cv_scores is not None:
            metrics["cv_accuracy_mean"] = cv_scores.mean()
            metrics["cv_accuracy_std"] = cv_scores.std()
    elif target == "margin":
        y_pred = model.predict(X_test)
        metrics["rmse"] = mean_squared_error(y_test, y_pred, squared=False)
        cv_scores = _safe_cross_val(model, X, y, scoring="neg_root_mean_squared_error")
        if cv_scores is not None:
            metrics["cv_rmse_mean"] = -cv_scores.mean()
            metrics["cv_rmse_std"] = cv_scores.std()
    elif target == "outcome_category":
        y_pred = model.predict(X_test)
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        cv_scores = _safe_cross_val(model, X, y, scoring="accuracy")
        if cv_scores is not None:
            metrics["cv_accuracy_mean"] = cv_scores.mean()
            metrics["cv_accuracy_std"] = cv_scores.std()

    # Save model artifact
    artifact = {"model": model, "features": list(X.columns), "target": target}
    joblib.dump(artifact, MODEL_FILE_PKL)
    logger.info(f"✅ Team model trained on target='{target}' and saved to {MODEL_FILE_PKL}")

    # Save metrics
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_df = pd.DataFrame([metrics])
    metrics_file = os.path.join(RESULTS_DIR, f"metrics_team_{target}_{model_type}.csv")
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"📊 Team metrics saved to {metrics_file}: {metrics}")


def train_player_model(target="pts", model_type="linear"):
    if not os.path.exists(PLAYER_FEATURES_FILE):
        raise DataError(f"Player features file not found: {PLAYER_FEATURES_FILE}")

    df = pd.read_csv(PLAYER_FEATURES_FILE)
    logger.info(f"📂 Loaded player dataset with shape {df.shape}")

    if target not in df.columns:
        raise DataError