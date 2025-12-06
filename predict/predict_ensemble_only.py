# ============================================================
# File: predict/predict_ensemble_only.py
# Purpose: Predict games using ONLY the ensemble model
# ============================================================

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import datetime

# --- Project Root Path Fix ---
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

logger = init_global_logger()


# ============================================================
# Helpers
# ============================================================
def safe_load(path: Path):
    """Load model safely, return None if missing/broken."""
    if path.exists():
        try:
            return joblib.load(path)
        except Exception as e:
            logger.error(f"❌ Failed to load model {path}: {e}")
    return None


def aggregate_player_predictions(team_df, player_df, player_model):
    """
    Convert player-level predictions to game-level (mean aggregated),
    matching logic used in train_ensemble.py.
    """

    if player_df.empty or player_model is None:
        return pd.Series([0.0] * len(team_df), index=team_df.index)

    # Join keys required for alignment
    join_keys = ["game_id", "home_team", "away_team"]
    join_keys = [c for c in join_keys if c in team_df.columns and c in player_df.columns]

    if not join_keys:
        logger.warning("⚠️ Player data cannot align to team rows — using zeros")
        return pd.Series([0.0] * len(team_df), index=team_df.index)

    # Clean player feature set
    clean_player = player_df.drop(columns=["player_name"], errors="ignore")
    feat_cols = clean_player.columns.difference(join_keys)

    if feat_cols.empty:
        logger.warning("⚠️ Player data lacks numeric features — using zeros")
        return pd.Series([0.0] * len(team_df), index=team_df.index)

    try:
        if hasattr(player_model, "predict_proba"):
            clean_player["pred"] = player_model.predict_proba(clean_player[feat_cols])[:, 1]
        else:
            clean_player["pred"] = player_model.predict(clean_player[feat_cols])

    except Exception as e:
        logger.error(f"❌ Player prediction failed: {e}")
        return pd.Series([0.0] * len(team_df), index=team_df.index)

    # Aggregate to game-level
    grouped = clean_player.groupby(join_keys)["pred"].mean().reset_index()
    merged = team_df.merge(grouped, on=join_keys, how="left")

    merged["pred"].fillna(0.0, inplace=True)
    return merged["pred"]


# ============================================================
# Ensemble-only prediction
# ============================================================
def predict_ensemble_only():
    ensure_dirs()

    if not NEW_GAMES_FEATURES_FILE.exists():
        logger.error(f"❌ Missing new features file: {NEW_GAMES_FEATURES_FILE}")
        return pd.DataFrame()

    df = pd.read_csv(NEW_GAMES_FEATURES_FILE)
    if df.empty:
        logger.warning("⚠️ No new games available.")
        return pd.DataFrame()

    # Load ensemble model
    ensemble_model = safe_load(ENSEMBLE_MODEL_FILE)
    if ensemble_model is None:
        logger.error("❌ Ensemble model missing — cannot continue.")
        return pd.DataFrame()

    # Load base models (team, ml, ou, player)
    team_model   = safe_load(TEAM_MODEL_FILE)
    ml_model     = safe_load(XGB_ML_MODEL_FILE)
    ou_model     = safe_load(XGB_OU_MODEL_FILE)
    player_model = safe_load(PLAYER_MODEL_FILE)

    # Build base prediction feature set
    X_team = df.drop(columns=["game_id", "home_team", "away_team"], errors="ignore")
    preds = {}

    # Team model
    if team_model:
        try:
            preds["team"] = team_model.predict_proba(X_team)[:, 1]
        except Exception as e:
            logger.warning(f"⚠️ Team model failed: {e}")

    # ML model
    if ml_model:
        try:
            preds["ml"] = ml_model.predict_proba(X_team)[:, 1]
        except Exception as e:
            logger.warning(f"⚠️ ML model failed: {e}")

    # OU model
    if ou_model:
        try:
            preds["ou"] = ou_model.predict_proba(X_team)[:, 1]
        except Exception as e:
            logger.warning(f"⚠️ OU model failed: {e}")

    # Player model → aggregated
    player_df_path = Path("data/raw/player_games.csv")
    if player_df_path.exists() and player_model:
        try:
            player_df = pd.read_csv(player_df_path)
            preds["player"] = aggregate_player_predictions(df, player_df, player_model)
        except Exception as e:
            logger.warning(f"⚠️ Player model aggregation failed: {e}")

    if not preds:
        logger.error("❌ No base predictions available for ensemble inference.")
        return pd.DataFrame()

    # Assemble meta-feature matrix
    ensemble_X = pd.DataFrame(preds).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Ensemble predict
    try:
        df["home_win_prob"] = ensemble_model.predict_proba(ensemble_X)[:, 1]
        df["prediction"] = (df["home_win_prob"] >= 0.5).astype(int)
        logger.info("🔮 Ensemble model successfully generated predictions")
    except Exception as e:
        logger.error(f"❌ Ensemble prediction failed: {e}")
        return pd.DataFrame()

    # Save output
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"ensemble_predictions_{ts}.csv"

    df_out = df[["game_id", "home_team", "away_team", "home_win_prob", "prediction"]]
    df_out.to_csv(out_file, index=False)

    logger.info(f"📂 Ensemble predictions saved → {out_file}")

    return df_out


# ============================================================
# Entrypoint
# ============================================================
def main():
    df = predict_ensemble_only()
    if df is not None and not df.empty:
        print(df)


if __name__ == "__main__":
    main()
