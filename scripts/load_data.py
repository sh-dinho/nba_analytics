import pandas as pd
import logging
from core.db_module import connect, init_db

logging.basicConfig(level=logging.INFO)

def scrape_season(season: int) -> pd.DataFrame:
    """
    Scrape NBA season schedule/results from Basketball Reference.
    Includes both past results and future scheduled games.
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    tables = pd.read_html(url)
    df = tables[0]  # first table is the schedule/results

    # Normalize columns
    df.rename(columns={
        "Date": "date",
        "Visitor/Neutral": "away_team",
        "PTS": "away_score",
        "Home/Neutral": "home_team",
        "PTS.1": "home_score"
    }, inplace=True)

    # Add season + game_id
    df["season"] = season
    df["game_id"] = df.index.astype(str) + "_" + df["season"].astype(str)

    # Winner logic (only for completed games)
    def winner(row):
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            return None  # scheduled but not played yet
        return row["home_team"] if row["home_score"] > row["away_score"] else row["away_team"]

    df["winner"] = df.apply(winner, axis=1)

    return df[["game_id", "date", "season", "home_team", "away_team", "home_score", "away_score", "winner"]]

def load_season_to_db(season: int):
    """Scrape and load one season into the nba_games table."""
    df = scrape_season(season)
    init_db()
    with connect() as con:
        cur = con.cursor()
        for _, row in df.iterrows():
            cur.execute("""
                INSERT OR REPLACE INTO nba_games
                (game_id, date, season, home_team, away_team, home_score, away_score, winner)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["game_id"],
                row["date"],
                int(row["season"]),
                row["home_team"],
                row["away_team"],
                int(row["home_score"]) if not pd.isna(row["home_score"]) else None,
                int(row["away_score"]) if not pd.isna(row["away_score"]) else None,
                row["winner"]
            ))
        con.commit()
    logging.info(f"✔ Loaded {len(df)} games (played + scheduled) for season {season}")

if __name__ == "__main__":
    # Example: scrape and load multiple seasons automatically
    for season in [2023, 2024, 2025]:
        load_season_to_db(season)