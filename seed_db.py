# seed_db.py
import sqlite3
import logging
from datetime import datetime, timedelta
import yaml
from db_module import init_db

logging.basicConfig(level=logging.INFO)

try:
    with open("config.yaml") as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    raise RuntimeError("⚠️ config.yaml not found. Please create one with database.path")

DB_PATH = CONFIG["database"]["path"]

def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def seed_db(days: int = 40):
    now = datetime.now()
    init_db()

    with connect() as con:
        cur = con.cursor()

        # Seed picks (idempotent)
        cur.executemany("""
        INSERT OR IGNORE INTO daily_picks (Timestamp, home_team, away_team, ev)
        VALUES (?,?,?,?)
        """, [
            ((now - timedelta(days=2)).isoformat(), "Lakers", "Celtics", 0.12),
            ((now - timedelta(days=1)).isoformat(), "Warriors", "Spurs", -0.05),
            (now.isoformat(), "Raptors", "Heat", 0.08),
        ])

        # Seed bankroll (last N days)
        rows = []
        base = 1000.0
        for i in range(days):
            ts = (now - timedelta(days=i)).isoformat()
            bankroll = base + i * 5 - (i % 7) * 3
            roi = (bankroll - base) / base
            rows.append((ts, bankroll, roi))
        cur.executemany("""
        INSERT OR IGNORE INTO bankroll_tracker (Timestamp, CurrentBankroll, ROI)
        VALUES (?,?,?)
        """, rows)

        # Seed model metrics
        cur.executemany("""
        INSERT OR IGNORE INTO model_metrics (Timestamp, AUC, Accuracy)
        VALUES (?,?,?)
        """, [
            (now.isoformat(), 0.725, 0.665),
            ((now - timedelta(days=3)).isoformat(), 0.711, 0.651),
        ])

        # Seed retrain history
        cur.executemany("""
        INSERT OR IGNORE INTO retrain_history (Timestamp, ModelType, Status)
        VALUES (?,?,?)
        """, [
            ((now - timedelta(days=6)).isoformat(), "XGBoost", "Success"),
            ((now - timedelta(days=14)).isoformat(), "RandomForest", "Failed"),
        ])

        con.commit()
    logging.info("✔ Seeded database with test data")

if __name__ == "__main__":
    seed_db()