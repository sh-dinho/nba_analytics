import os
import joblib
import yaml
import numpy as np
import pandas as pd
import logging

ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.yaml")
CONFIG = yaml.safe_load(open(CONFIG_PATH))

MODELS_DIR = CONFIG["model"]["models_dir"]
ENCODER_PATH = os.path.join(MODELS_DIR, "team_encoder.pkl")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ============================================================
# LOAD MOST RECENT MODEL
# ============================================================
def load_latest_model():
    """Load the most recent XGBoost model from models/ directory."""
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl") and f.startswith("xgb_model_")]

    if not models:
        raise FileNotFoundError("❌ No trained model found in models directory.")

    # Sort by version timestamp
    models.sort(reverse=True)

    latest_file = models[0]
    full_path = os.path.join(MODELS_DIR, latest_file)

    model = joblib.load(full_path)
    version = latest_file.replace(".pkl", "")

    logging.info(f"✅ Loaded model: {latest_file}")

    return model, version


# ============================================================
# LOAD TEAM ENCODER
# ============================================================
def load_team_encoder():
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError("❌ Missing team_encoder.pkl — run training first.")

    return joblib.load(ENCODER_PATH)


# ============================================================
# PREPARE INPUT FOR MODEL
# ============================================================
def prepare_features(home_team, visitor_team):
    """Convert team names to numeric encoded features."""
    encoder = load_team_encoder()

    if home_team not in encoder["team_to_id"] or visitor_team not in encoder["team_to_id"]:
        raise ValueError(f"Unknown team(s): {home_team}, {visitor_team}")

    h_id = encoder["team_to_id"][home_team]
    v_id = encoder["team_to_id"][visitor_team]

    df = pd.DataFrame([{
        "home_id": h_id,
        "visitor_id": v_id
    }])

    return df


# ============================================================
# PREDICT MATCHUP
# ============================================================
def predict_matchup(model, home_team, visitor_team):
    """Return probability that the HOME team wins."""
    X = prepare_features(home_team, visitor_team)

    prob_home_win = model.predict_proba(X)[0][1]  # Class "1" = home win

    return float(prob_home_win)
