# ============================================================
# File: predict/predict.py
# Purpose: Ensemble stacking with team+player base models, meta-learner, and SHAP explainability
# ============================================================

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from nba_core.log_config import init_global_logger
from nba_core.paths import RESULTS_DIR
from nba_core.config import ENSEMBLE_MODEL_FILE, TEAM_MODEL_FILE, PLAYER_MODEL_FILE

logger = init_global_logger("predict")

def predict(features_path: Path, return_shap: bool = False):
    df = pd.read_csv(features_path)
    # Select only numeric features for models; adapt as needed
    X = df.select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("No numeric features found for prediction")

    # Load models
    team_model = joblib.load(TEAM_MODEL_FILE)
    player_model = joblib.load(PLAYER_MODEL_FILE)
    ensemble_model = joblib.load(ENSEMBLE_MODEL_FILE)

    # Base predictions
    team_probs = team_model.predict_proba(X)[:, 1]
    player_probs = player_model.predict_proba(X)[:, 1]

    # Stack features
    stacked = np.vstack([team_probs, player_probs]).T

    # Meta-learner prediction
    final_probs = ensemble_model.predict_proba(stacked)[:, 1]

    df_out = pd.DataFrame({
        "home_team": df.get("home_team", pd.Series([None] * len(df))),
        "away_team": df.get("away_team", pd.Series([None] * len(df))),
        "prob": final_probs,                    # unified probability for bankroll sim
        "american_odds": df.get("american_odds", pd.Series([None] * len(df))),  # join real odds upstream when available
        "ou_prediction": np.where(final_probs > 0.5, "OVER", "UNDER")
    })
    logger.info(f"✅ Ensemble produced {len(df_out)} predictions")

    shap_path = None
    if return_shap:
        try:
            explainer = shap.Explainer(ensemble_model, stacked)
            shap_values = explainer(stacked)
            shap.summary_plot(shap_values, stacked, show=False)
            shap_path = RESULTS_DIR / "ensemble/ensemble_shap_summary.png"
            shap_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(shap_path)
            plt.close()
            logger.info(f"📊 SHAP summary saved → {shap_path}")
        except Exception as e:
            logger.warning(f"⚠️ SHAP generation failed: {e}")
            shap_path = None

    return (df_out, shap_path) if return_shap else df_out
