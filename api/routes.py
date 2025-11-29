from flask import Blueprint, jsonify
import subprocess
import yaml
import sqlite3
from datetime import datetime

bp = Blueprint("train", __name__)
CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]

@bp.route("/train/run", methods=["POST"])
def retrain_model():
    try:
        subprocess.run(["python", "train/train_model_xgb.py"], check=True)
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("""
            INSERT INTO retrain_history (Timestamp, ModelType, Status)
            VALUES (?, ?, ?)
        """, (datetime.now().strftime("%Y-%m-%d %H:%M"), "xgboost", "triggered_via_api"))
        con.commit()
        con.close()
        return jsonify({"status": "ok", "message": "Model retraining started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@bp.route("/metrics/retrain", methods=["GET"])
def retrain_metrics():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT Timestamp, ModelType, Status FROM retrain_history ORDER BY Timestamp DESC LIMIT 20")
    rows = cur.fetchall()
    con.close()
    history = [{"Timestamp": r[0], "ModelType": r[1], "Status": r[2]} for r in rows]
    return jsonify({"history": history})

@bp.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "time": datetime.now().strftime("%Y-%m-%d %H:%M")}
