# ============================================================
# File: predict/predict_ensemble_only.py
# Purpose: Generate predictions strictly using the ensemble model
# ============================================================

import sys
from pathlib import Path
import pandas as pd
import joblib
import datetime

# --- Project Path Fix ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Imports ---
from core.paths import (
    NEW_GAMES_FEATURES_FILE,
    ENSEMBLE_MODEL_FILE,
    TEAM_MODEL_FILE,
    PLAYER_MODEL_FILE,
    XGB_ML_MODEL_FILE,
    XGB_OU_MODEL_FILE,
    RESULTS_DIR,
    ensure_dirs,
)
from core.log_config import init_global_logger
from core.exceptions import DataError

logger = init_global_logger()

# ---------------- Safe Model Loader ----------------
def safe_load_model(path: Path):
    if not path.exists():
        raise DataError(f"Required model missing: {path.name}")
    try:
        return joblib.load(path)
    except Exception as e:
        raise DataError(f"Failed to load model {path.name}: {e}") from e

# ---------------- Ensemble-only Prediction ----------------
def predict_ensemble_only():
    ensure_dirs(strict=False)

    if not NEW_GAMES_FEATURES_FILE.exists():
        raise DataError(f"Missing new game features: {NEW_GAMES_FEATURES_FILE}")

    df = pd.read_csv(NEW_GAMES_FEATURES_FILE)
    if df.empty:
        raise DataError("New games feature file is empty. No predictions can be generated.")

    # Load ensemble model
    ensemble_model = safe_load_model(ENSEMBLE_MODEL_FILE)
    logger.info("🔮 Using ensemble model for predictions")

    # Load base models required for ensemble
    base_models = {
        "team": safe_load_model(TEAM_MODEL_FILE),
        "ml": safe_load_model(XGB_ML_MODEL_FILE),
        "ou": safe_load_model(XGB_OU_MODEL_FILE),
        "player": safe_load_model(PLAYER_MODEL_FILE)
    }

    X = df.drop(columns=["game_id", "home_team", "away_team"], errors="ignore")
    preds = {}

    for name, model in base_models.items():
        try:
            if hasattr(model, "predict_proba"):
                preds[name] = model.predict_proba(X)[:, 1]
            else:
                preds[name] = model.predict(X)
        except Exception as e:
            raise DataError(f"Failed to generate predictions from {name} model: {e}") from e

    if not preds:
        raise DataError("No base model predictions available. Ensemble cannot be used.")

    # Build ensemble feature dataframe and predict
    ensemble_X = pd.DataFrame(preds)
    try:
        df["prediction"] = ensemble_model.predict(ensemble_X)
    except Exception as e:
        raise DataError(f"Ensemble prediction failed: {e}") from e

    # Save predictions
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"predictions_ensemble_only_{timestamp}.csv"
    df_to_save = df[["game_id", "home_team", "away_team", "prediction"]].copy()
    df_to_save.to_csv(out_file, index=False)
    logger.info(f"📂 Ensemble-only predictions saved → {out_file}")

    return df_to_save

# ---------------- Entrypoint ----------------
def main():
    results = predict_ensemble_only()
    print(results)

if __name__ == "__main__":
    main()
