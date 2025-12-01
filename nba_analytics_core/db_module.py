# nba_analytics_core/db_module.py
import sqlite3
import logging
from typing import Iterable, Dict, Any, List, Tuple
from config import DB_PATH

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    season INTEGER,
    date TEXT,
    home_team TEXT,
    away_team TEXT,
    home_score INTEGER,
    away_score INTEGER
);

CREATE TABLE IF NOT EXISTS features (
    game_id TEXT PRIMARY KEY,
    home_win INTEGER,
    total_points INTEGER,
    FOREIGN KEY (game_id) REFERENCES games (game_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    game_id TEXT PRIMARY KEY,
    predicted_winner TEXT,
    predicted_prob REAL,
    created_at TEXT,
    FOREIGN KEY (game_id) REFERENCES games (game_id)
);
"""

def get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)

def init_db() -> None:
    logging.info(f"Initializing database at {DB_PATH}")
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)
    logging.info("✔ Database initialized")

def insert_games(rows: Iterable[Dict[str, Any]]) -> int:
    with get_conn() as conn:
        cur = conn.cursor()
        inserted = 0
        for r in rows:
            try:
                cur.execute(
                    "INSERT OR REPLACE INTO games (game_id, season, date, home_team, away_team, home_score, away_score) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        r["game_id"],
                        r["season"],
                        r.get("date"),
                        r["home_team"],
                        r["away_team"],
                        r.get("home_score"),
                        r.get("away_score"),
                    ),
                )
                inserted += 1
            except Exception:
                logging.exception(f"Failed to insert game {r.get('game_id')}")
        conn.commit()
        logging.info(f"✔ Inserted {inserted} games")
        return inserted

def insert_features(rows: Iterable[Dict[str, Any]]) -> int:
    with get_conn() as conn:
        cur = conn.cursor()
        inserted = 0
        for r in rows:
            try:
                cur.execute(
                    "INSERT OR REPLACE INTO features (game_id, home_win, total_points) VALUES (?, ?, ?)",
                    (
                        r["game_id"],
                        int(bool(r["home_win"])) if r.get("home_win") is not None else None,
                        r.get("total_points"),
                    ),
                )
                inserted += 1
            except Exception:
                logging.exception(f"Failed to insert features {r.get('game_id')}")
        conn.commit()
        logging.info(f"✔ Inserted {inserted} feature rows")
        return inserted

def load_season_to_db(season: int) -> None:
    # Placeholder: fetch season data; integrate real scraper here
    logging.info(f"Loading season {season} into DB...")
    sample_games = [
        {"game_id": f"{season}-001", "season": season, "date": f"{season}-10-20", "home_team": "LAL", "away_team": "BOS", "home_score": 110, "away_score": 108},
        {"game_id": f"{season}-002", "season": season, "date": f"{season}-10-21", "home_team": "NYK", "away_team": "BOS", "home_score": 95, "away_score": 100},
    ]
    insert_games(sample_games)
    feature_rows = [
        {"game_id": g["game_id"], "home_win": (g["home_score"] or 0) > (g["away_score"] or 0), "total_points": (g["home_score"] or 0) + (g["away_score"] or 0)}
        for g in sample_games
    ]
    insert_features(feature_rows)

def export_feature_stats() -> None:
    logging.info("Exporting feature statistics...")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), AVG(total_points) FROM features")
        count, avg_total = cur.fetchone()
        logging.info(f"✔ Feature stats: rows={count}, avg_total_points={avg_total}")