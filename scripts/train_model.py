# ============================================================
# File: scripts/train_model.py
# Purpose: Train predictive models on NBA features with optional hyperparameter tuning
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
from xgboost import XGBClassifier
from core.config import TRAINING_FEATURES_FILE, MODEL_FILE_PKL, RESULTS_DIR
from core.log_config import setup_logger
from core.exceptions import DataError

logger = setup_logger("train_model")


def main(target: str = "label", model_type: str = "logistic", tune: bool = False):
    if not os.path.exists(TRAINING_FEATURES_FILE):
        raise DataError(f"Training features file not found: {TRAINING_FEATURES_FILE}")

    df = pd.read_csv(TRAINING_FEATURES_FILE)

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
    X = df[feature_cols]
    y = df[target]

    # Encode target if necessary
    if target == "label" and y.dtype == "object":
        y = y.map({"HOME": 1, "AWAY": 0})

    # For XGBoost, ensure all features are numeric
    if model_type == "xgb":
        X = pd.get_dummies(X, drop_first=True)
        X = X.fillna(0)

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

            if tune:
                # Hyperparameter search space
                param_dist = {
                    "n_estimators": [200, 500, 1000],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0]
                }

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

                # Run search WITHOUT early stopping
                search.fit(X_train, y_train)

                # Get best model and refit WITH early stopping
                best_model = search.best_estimator_
                logger.info(f"✅ Best XGBoost params: {search.best_params_}")

                best_model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                model = best_model
            else:
                # Normal fit with early stopping
                base_model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                model = base_model
        else:
            # Logistic regression with preprocessing
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
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        metrics["log_loss"] = log_loss(y_test, y_prob)
        metrics["brier"] = brier_score_loss(y_test, y_prob)
        metrics["auc"] = roc_auc_score(y_test, y_prob)

        cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        metrics["cv_accuracy_mean"] = cv_scores.mean()
        metrics["cv_accuracy_std"] = cv_scores.std()

    elif target == "margin":
        y_pred = model.predict(X_test)
        metrics["rmse"] = mean_squared_error(y_test, y_pred, squared=False)

        cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
        metrics["cv_rmse_mean"] = -cv_scores.mean()
        metrics["cv_rmse_std"] = cv_scores.std()

    elif target == "outcome_category":
        y_pred = model.predict(X_test)
        metrics["accuracy"] = accuracy_score(y_test, y_pred)

        cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        metrics["cv_accuracy_mean"] = cv_scores.mean()
        metrics["cv_accuracy_std"] = cv_scores.std()

    # Save model artifact
    artifact = {"model": model, "features": list(X.columns), "target": target}
    joblib.dump(artifact, MODEL_FILE_PKL)
    logger.info(f"✅ Model trained on target='{target}' and saved to {MODEL_FILE_PKL}")

    # Save metrics
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_df = pd.DataFrame([metrics])
    metrics_file = os.path.join(RESULTS_DIR, "training_metrics.csv")
    if os.path.exists(metrics_file):
        metrics_df.to_csv(metrics_file, mode="a", header=False, index=False)
    else:
        metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"📊 Metrics saved to {metrics_file}: {metrics}")

    # Save feature importance if available
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": list(X.columns),
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        importance_file = os.path.join(RESULTS_DIR, "feature_importance.csv")
        importance_df.to_csv(importance_file, index=False)
        logger.info(f"📈 Feature importances saved to {importance_file}")

    # Save logistic regression coefficients if available
    if isinstance(model, Pipeline) and "logreg" in model.named_steps:
        logreg = model.named_steps["logreg"]
        coef_df = pd.DataFrame({
            "feature": list(X.columns),
            "coefficient": logreg.coef_[0]
        }).sort_values("coefficient", ascending=False)
        coef_file = os.path.join(RESULTS_DIR, "logreg_coefficients.csv")
        coef_df.to_csv(coef_file, index=False)
        logger.info(f"📉 Logistic regression coefficients saved to {coef_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NBA prediction model")
    parser.add_argument("--target", type=str, default="label",
                        help="Target column: label, margin, outcome_category")
    parser.add_argument("--model_type", type=str, default="logistic",
                        help="Model type: logistic, rf, xgb, linear")
    parser.add_argument("--tune", action="store_true",
                        help="Enable hyperparameter tuning for XGBoost")
    args = parser.parse_args()

    main(target=args.target, model_type=args.model_type, tune=args.tune)