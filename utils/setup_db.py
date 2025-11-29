import sqlite3

DB_PATH = "nba_analytics.db"

def create_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS nba_games (
        GAME_ID TEXT,
        GAME_DATE TEXT,
        TEAM_ABBREVIATION TEXT,
        PTS REAL,
        REB REAL,
        AST REAL,
        WL TEXT,
        MATCHUP TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS daily_picks (
        Timestamp TEXT,
        Team TEXT,
        Opponent TEXT,
        Probability REAL,
        Odds REAL,
        EV REAL,
        SuggestedStake REAL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS bankroll_tracker (
        Timestamp TEXT,
        StartingBankroll REAL,
        CurrentBankroll REAL,
        ROI REAL,
        Notes TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS retrain_history (
        Timestamp TEXT,
        ModelType TEXT,
        Status TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS model_metrics (
        Timestamp TEXT,
        Accuracy REAL,
        AUC REAL
    )
    """)

    con.commit()
    con.close()
    print("✅ Database created successfully.")

if __name__ == "__main__":
    create_db()
