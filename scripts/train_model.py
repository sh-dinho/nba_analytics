# File: scripts/train_model.py
import os
import json
import random
import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss, roc_auc_score
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from lightgbm import LGBMClassifier
import shap
import matplotlib.pyplot as plt
import joblib

# ----------------------------
# Logging setup
# ----------------------------
logger = logging.getLogger("train_model")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def main():
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv("data/training_features.csv")

    # ----------------------------
    # Validate target
    ----------------------------
    if "home_win" not in df.columns or df["home_win"].isna().all():
        raise ValueError("Target home_win missing or empty in training data.")

    target = df["home_win"].astype(int)

    # ----------------------------
    # Feature selection
    # (exclude target explicitly)
    # ----------------------------
    feature_cols = [
        c for c in df.columns
        if (c.startswith("home_") or c.startswith("away_"))
        and c != "home_win"
    ]

    if len(feature_cols) == 0:
        raise ValueError("No features found. Check your feature generation pipeline.")

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())  # safer than filling with 0

    # ----------------------------
    # Train/validation split
    # ----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, target, test_size=0.2, random_state=SEED, stratify=target
    )

    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    logger.info(f"Class balance: train positive={y_train.mean():.3f}, val positive={y_val.mean():.3f}")

    # ----------------------------
    # Base LightGBM model (unfitted)
    # ----------------------------
    base_model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=0.1,
        random_state=SEED
    )

    # ----------------------------
    # Probability calibration (isotonic)
    # CalibratedClassifierCV will fit base_model internally
    # ----------------------------
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)

    # ----------------------------
    # Metrics
    # ----------------------------
    proba = calibrated.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_val, preds)),
        "log_loss": float(log_loss(y_val, proba)),
        "brier": float(brier_score_loss(y_val, proba)),
        "auc": float(roc_auc_score(y_val, proba))
    }

    # ----------------------------
    # Calibration curve
    # ----------------------------
    prob_true, prob_pred = calibration_curve(y_val, proba, n_bins=10)

    metrics["calibration_curve"] = {
        "mean_pred": [float(x) for x in prob_pred],
        "frac_pos": [float(x) for x in prob_true]
    }

    plt.figure()
    plt.plot(prob_pred, prob_true, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.savefig("models/calibration_curve.png")
    plt.close()
    logger.info("📈 Calibration curve saved to models/calibration_curve.png")

    # ----------------------------
    # Save model
    # ----------------------------
    joblib.dump(calibrated, "models/game_predictor.pkl")
    ts_model = f"models/game_predictor_{_timestamp()}.pkl"
    joblib.dump(calibrated, ts_model)

    logger.info("✅ Model saved to models/game_predictor.pkl")
    logger.info(f"📦 Timestamped backup saved to {ts_model}")

    # ----------------------------
    # Metadata
    # ----------------------------
    meta = {
        "model_type": "LightGBM + Isotonic calibration",
        "n_features": len(feature_cols),
        "features": feature_cols,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }

    # ----------------------------
    # SHAP Feature Importance
    # ----------------------------
    try:
        # Extract trained underlying estimator
        fitted_model = calibrated.calibrated_classifiers_[0].estimator

        explainer = shap.TreeExplainer(fitted_model)

        # Use SHAP on a limited sample if large
        sample = X_val.sample(n=min(200, len(X_val)), random_state=SEED)

        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):  # multiclass guard
            shap_values = shap_values[1]

        plt.figure()
        shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig("models/shap_feature_importance.png")
        plt.close()
        logger.info("📈 SHAP feature importance saved to models/shap_feature_importance.png")

        # Compute mean absolute SHAP values
        shap_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": sample.columns,
            "mean_abs_shap": shap_importance
        }).sort_values("mean_abs_shap", ascending=False)

        meta["top_features"] = importance_df.head(10).to_dict(orient="records")

    except Exception as e:
        logger.warning(f"⚠️ SHAP analysis failed: {e}")

    # ----------------------------
    # Save metadata
    # ----------------------------
    meta_file = "models/model_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    ts_meta = f"models/model_metadata_{_timestamp()}.json"
    with open(ts_meta, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"🧾 Model metadata saved to {meta_file}")
    logger.info(f"📦 Timestamped metadata backup saved to {ts_meta}")

    logger.info(
        f"📊 Metrics: acc={metrics['accuracy']:.3f}, "
        f"logloss={metrics['log_loss']:.3f}, "
        f"brier={metrics['brier']:.3f}, "
        f"auc={metrics['auc']:.3f}"
    )
    if "top_features" in meta:
        logger.info("📊 Top feature importances included in model_metadata.json")

    return metrics


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        sys.exit(1)
