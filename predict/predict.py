# ============================================================
# File: predict/predict.py
# Purpose: Generate predictions using trained models
# ============================================================

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import datetime

# --- Project Path Fix ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Imports ---
from core.paths import (
    NEW_GAMES_FEATURES_FILE,
    TEAM_MODEL_FILE,
    PLAYER_MODEL_FILE,
    XGB_ML_MODEL_FILE,
    XGB_OU_MODEL_FILE,
    ENSEMBLE_MODEL_FILE,
    RESULTS_DIR,
    ensure_dirs,
)
from core.log_config import init_global_logger
from core.exceptions import DataError

logger = init_global_logger()


# ============================================================
# Helper: Safe model loading
# ============================================================
def safe_load(path: Path):
    if path.exists():
        try:
            return joblib.load(path)
        except Exception as e:
            logger.warning(f"⚠️ Failed to load model {path}: {e}")
    return None


# ============================================================
# Helper: Aggregate player predictions to team rows
# ============================================================
def aggregate_player_predictions(team_df, player_df, player_model):
    """
    Convert player-level predictions → single game-level feature.
    Matches the logic used in train_ensemble.py.
    """
    if player_df.empty or player_model is None:
        return pd.Series([0.0] * len(team_df), index=team_df.index)

    # Determine join keys that exist in both datasets
    game_keys = ["game_id", "home_team", "away_team"]
    game_keys = [c for c in game_keys if c in team_df.columns and c in player_df.columns]

    if not game_keys:
        logger.warning("⚠️ Cannot align player data to team rows. Using zeros.")
        return pd.Series([0.0] * len(team_df), index=team_df.index)

    # Prepare player features
    clean_player = player_df.drop(columns=["player_name"], errors="ignore")

    feat_cols = clean_player.columns.difference(game_keys)
    if feat_cols.empty:
        logger.warning("⚠️ Player data has no usable numeric features.")
        return pd.Series([0.0] * len(team_df), index=team_df.index)

    # Predict per-player performance
    try:
        if hasattr(player_model, "predict_proba"):
            clean_player["pred"] = player_model.predict_proba(clean_player[feat_cols])[:, 1]
        else:
            clean_player["pred"] = player_model.predict(clean_player[feat_cols])
    except Exception as e:
        logger.warning(f"⚠️ Player prediction failed: {e}")
        return pd.Series([0.0] * len(team_df), index=team_df.index)

    # Aggregate per game
    grouped = clean_player.groupby(game_keys)["pred"].mean().reset_index()

    merged = team_df.merge(grouped, on=game_keys, how="left")
    merged["pred"].fillna(0.0, inplace=True)

    return merged["pred"]


# ============================================================
# Main Prediction Flow
# ============================================================
def predict():
    ensure_dirs()

    if not NEW_GAMES_FEATURES_FILE.exists():
        logger.warning(f"⚠️ Missing new game features: {NEW_GAMES_FEATURES_FILE}")
        return pd.DataFrame()

    df = pd.read_csv(NEW_GAMES_FEATURES_FILE)
    if df.empty:
        logger.warning("⚠️ No new games found.")
        return pd.DataFrame()

    # Load all models
    ensemble_model = safe_load(ENSEMBLE_MODEL_FILE)
    team_model    = safe_load(TEAM_MODEL_FILE)
    ml_model      = safe_load(XGB_ML_MODEL_FILE)
    ou_model      = safe_load(XGB_OU_MODEL_FILE)
    player_model  = safe_load(PLAYER_MODEL_FILE)

    # Prepare base feature frame (same as ensemble training)
    X_team = df.drop(columns=["game_id", "home_team", "away_team"], errors="ignore")

    base_preds = {}

    # TEAM
    if team_model:
        try:
            base_preds["team"] = team_model.predict_proba(X_team)[:, 1]
        except Exception as e:
            logger.warning(f"⚠️ Team model failed: {e}")

    # ML
    if ml_model:
        try:
            base_preds["ml"] = ml_model.predict_proba(X_team)[:, 1]
        except Exception as e:
            logger.warning(f"⚠️ ML model failed: {e}")

    # OU
    if ou_model:
        try:
            base_preds["ou"] = ou_model.predict_proba(X_team)[:, 1]
        except Exception as e:
            logger.warning(f"⚠️ OU model failed: {e}")

    # PLAYER (requires aggregated predictions)
    player_df_path = Path("data/raw/player_games.csv")
    if player_df_path.exists() and player_model:
        try:
            player_df = pd.read_csv(player_df_path)
            base_preds["player"] = aggregate_player_predictions(df, player_df, player_model)
        except Exception as e:
            logger.warning(f"⚠️ Player-level prediction failed: {e}")

    # Convert to meta-feature DataFrame
    ensemble_X = pd.DataFrame(base_preds).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # -----------------------------
    # Use ensemble model if available
    # -----------------------------
    if ensemble_model and not ensemble_X.empty:
        try:
            df["home_win_prob"] = ensemble_model.predict_proba(ensemble_X)[:, 1]
            df["prediction"] = (df["home_win_prob"] >= 0.5).astype(int)
            logger.info("🔮 Ensemble model used for predictions")
        except Exception as e:
            logger.error(f"❌ Ensemble prediction failed: {e}")
            df["home_win_prob"] = None
    else:
        logger.info("⚠️ Ensemble unavailable → Falling back to team model")

    # -----------------------------
    # Fallback to team model
    # -----------------------------
    if "home_win_prob" not in df.columns or df["home_win_prob"].isnull().all():
        if team_model:
            try:
                df["home_win_prob"] = team_model.predict_proba(X_team)[:, 1]
                df["prediction"] = (df["home_win_prob"] >= 0.5).astype(int)
                logger.info("🏀 Team model used for fallback predictions")
            except Exception as e:
                logger.error(f"❌ Team fallback failed: {e}")
                df["home_win_prob"] = None
                df["prediction"] = None
        else:
            logger.error("❌ No usable model available for predictions")
            df["home_win_prob"] = None
            df["prediction"] = None

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"predictions_{ts}.csv"

    df_out = df[["game_id", "home_team", "away_team", "home_win_prob", "prediction"]]
    df_out.to_csv(out_file, index=False)

    logger.info(f"📂 Saved predictions → {out_file}")

    return df_out


# -------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------
def main():
    df = predict()
    if df is not None and not df.empty:
        print(df)


if __name__ == "__main__":
    main()
