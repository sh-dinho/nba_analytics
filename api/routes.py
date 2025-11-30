from flask import Blueprint, jsonify, request
import sqlite3
import yaml
import os
import logging
from datetime import datetime
from train.train_model_xgb import train_xgb_model
from models.predict import load_latest_model, predict_matchup

bp = Blueprint("api", __name__)

# ------------------------------
# Load config
# ------------------------------
ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.yaml")
CONFIG = yaml.safe_load(open(CONFIG_PATH))

DB_PATH = CONFIG["database"]["path"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ============================================================
# HEALTH CHECK
# ============================================================
@bp.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "database_exists": os.path.exists(DB_PATH)
    }


# ============================================================
# MODEL RETRAINING
# ============================================================
@bp.route("/train/run", methods=["POST"])
def run_retrain():
    try:
        result = train_xgb_model()

        # log retrain event
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()

        cur.execute("""
            INSERT INTO retrain_history (Timestamp, ModelVersion, Status)
            VALUES (?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            result["model_version"],
            "success"
        ))

        con.commit()
        con.close()

        return jsonify({"status": "ok", "result": result})

    except Exception as e:
        logging.error(f"Retrain failed: {e}")

        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()

        cur.execute("""
            INSERT INTO retrain_history (Timestamp, ModelVersion, Status)
            VALUES (?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "N/A",
            f"error: {e}"
        ))

        con.commit()
        con.close()

        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================
# RETRAIN HISTORY
# ============================================================
@bp.route("/metrics/retrain", methods=["GET"])
def retrain_metrics():
    con = sqlite3.connect(DB_PATH)
    df = con.execute("SELECT * FROM retrain_history ORDER BY Timestamp DESC LIMIT 20")
    rows = df.fetchall()
    con.close()

    history = [
        {"Timestamp": r[0], "ModelVersion": r[1], "Status": r[2]}
        for r in rows
    ]

    return jsonify({"history": history})


# ============================================================
# GET LATEST MODEL VERSION
# ============================================================
@bp.route("/model/latest", methods=["GET"])
def latest_model():
    model, version = load_latest_model()
    return jsonify({"model_version": version})


# ============================================================
# FETCH RECENT GAMES
# ============================================================
@bp.route("/games/recent", methods=["GET"])
def fetch_recent_games():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM nba_games ORDER BY Date DESC LIMIT 50", con)
    con.close()

    return jsonify(df.to_dict(orient="records"))


# ============================================================
# PREDICT A MATCHUP
# ============================================================
@bp.route("/predict", methods=["POST"])
def predict_endpoint():

    body = request.json
    home = body.get("home")
    visitor = body.get("visitor")

    if not home or not visitor:
        return jsonify({"error": "Missing home or visitor team"}), 400

    try:
        model, version = load_latest_model()
        prob = predict_matchup(model, home, visitor)

        return jsonify({
            "home": home,
            "visitor": visitor,
            "prob_home_win": float(prob),
            "model_version": version
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500
