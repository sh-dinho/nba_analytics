import sqlite3
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def init_db(db_path: str):
    # Create folder if needed
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    logging.info("🛠 Initializing database...")

    # Games table
    c.execute("""
        CREATE TABLE IF NOT EXISTS nba_games (
            GameID TEXT,
            Date TEXT,
            Visitor TEXT,
            VisitorPts INTEGER,
            Home TEXT,
            HomePts INTEGER,
            Winner TEXT,
            Season INTEGER,
            UNIQUE(GameID)
        );
    """)

    # Picks
    c.execute("""
        CREATE TABLE IF NOT EXISTS daily_picks (
            Timestamp TEXT,
            Team TEXT,
            Opponent TEXT,
            Probability REAL,
            Odds REAL,
            EV REAL,
            SuggestedStake REAL
        );
    """)

    # Bankroll
    c.execute("""
        CREATE TABLE IF NOT_EXISTS bankroll_tracker (
            Timestamp TEXT,
            StartingBankroll REAL,
            CurrentBankroll REAL,
            ROI REAL,
            Notes TEXT
        );
    """)

    # Train history
    c.execute("""
        CREATE TABLE IF NOT EXISTS retrain_history (
            Timestamp TEXT,
            ModelVersion TEXT,
            Status TEXT
        );
    """)

    # Model metrics
    c.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            Timestamp TEXT,
            Accuracy REAL,
            AUC REAL,
            LogLoss REAL,
            ModelVersion TEXT
        );
    """)

    conn.commit()
    conn.close()

    logging.info("✅ Database initialized successfully")
