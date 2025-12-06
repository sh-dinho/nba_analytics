# ============================================================
# Train team/player/ml/ou models and save feature lists
# ============================================================

import sys
from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nba_core.paths import (
    TRAINING_FEATURES_FILE,
    TEAM_MODEL_FILE,
    PLAYER_MODEL_FILE,
    XGB_ML_MODEL_FILE,
    XGB_OU_MODEL_FILE,
    ensure_dirs,
)
from nba_core.config import (
    RANDOM_SEED,
    TEST_SIZE,
    EVAL_METRICS,
    DEFAULT_XGB_PARAMS,
    DEFAULT_LOGREG_PARAMS,
    log_config_snapshot,
)
from nba_core.log_config import init_global_logger

logger = init_global_logger("train")

class DataError(Exception):
    pass

metrics_map = {
    "accuracy": accuracy_score,
    "f1": f1_score,
    "roc_auc": roc_auc_score,
}

def evaluate_model(model, X_test, y_test):
    if X_test.empty or y_test.empty:
        logger.warning("Empty test set. Skipping evaluation.")
        return {}
    preds = model.predict(X_test)
    probs = None
    try:
        probs = model.predict_proba(X_test)[:, 1]
    except Exception:
        pass
    results = {}
    for m in EVAL_METRICS:
        func = metrics_map.get(m)
        if func:
            if m == "roc_auc" and probs is not None:
                score = func(y_test, probs)
            else:
                score = func(y_test, preds)
            results[m] = float(score)
            logger.info(f"{m.upper()} = {score:.3f}")
    return results

def load_csv_safe(file_path: Path):
    if not file_path.exists():
        raise DataError(f"Feature file missing: {file_path}")
    df = pd.read_csv(file_path)
    if df.empty:
        raise DataError(f"Feature file is empty: {file_path}")
    return df

def train_model(model_type="team"):
    ensure_dirs(strict=False)
    log_config_snapshot()

    df = load_csv_safe(TRAINING_FEATURES_FILE)

    if "label" not in df.columns:
        raise DataError("No target column 'label' found in training data")
    y_col = "label"

    # Drop non-feature columns commonly present
    X = df.drop(columns=[y_col, "game_id", "home_team", "away_team", "player_name"], errors="ignore")
    y = df[y_col]

    if X.empty:
        raise DataError("Training features are empty after dropping non-feature columns")

    # Encode categorical to numeric
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category").cat.codes

    # Impute numeric
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    if model_type == "team":
        model = XGBClassifier(**DEFAULT_XGB_PARAMS)
        save_path = TEAM_MODEL_FILE
    elif model_type == "player":
        model = LogisticRegression(**DEFAULT_LOGREG_PARAMS)
        save_path = PLAYER_MODEL_FILE
    elif model_type == "ml":
        model = XGBClassifier(**DEFAULT_XGB_PARAMS)
        save_path = XGB_ML_MODEL_FILE
    elif model_type == "ou":
        model = XGBClassifier(**DEFAULT_XGB_PARAMS)
        save_path = XGB_OU_MODEL_FILE
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    logger.info(f"✅ {model_type.capitalize()} model trained on {len(X_train)} rows")

    results = evaluate_model(model, X_test, y_test)

    joblib.dump(model, save_path)
    logger.info(f"📂 {model_type.capitalize()} model saved → {save_path}")

    # Save feature list
    feature_file = save_path.with_suffix(".features.json")
    with open(feature_file, "w") as f:
        json.dump(list(X.columns), f)
    logger.info(f"📝 Feature list saved → {feature_file}")

    return results

def main():
    for mtype in ["team", "player", "ml", "ou"]:
        train_model(mtype)
    logger.info("✅ All models trained successfully")

if __name__ == "__main__":
    main()
