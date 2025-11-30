import sqlite3
import pandas as pd
import yaml
import logging

logging.basicConfig(level=logging.INFO)
CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]

CONFERENCE_MAP = {
    "Boston Celtics": "Eastern", "Brooklyn Nets": "Eastern", "New York Knicks": "Eastern", "Philadelphia 76ers": "Eastern", "Toronto Raptors": "Eastern",
    "Chicago Bulls": "Eastern", "Cleveland Cavaliers": "Eastern", "Detroit Pistons": "Eastern", "Indiana Pacers": "Eastern", "Milwaukee Bucks": "Eastern",
    "Atlanta Hawks": "Eastern", "Charlotte Hornets": "Eastern", "Miami Heat": "Eastern", "Orlando Magic": "Eastern", "Washington Wizards": "Eastern",
    "Dallas Mavericks": "Western", "Houston Rockets": "Western", "Memphis Grizzlies": "Western", "New Orleans Pelicans": "Western", "San Antonio Spurs": "Western",
    "Denver Nuggets": "Western", "Minnesota Timberwolves": "Western", "Oklahoma City Thunder": "Western", "Portland Trail Blazers": "Western", "Utah Jazz": "Western",
    "Golden State Warriors": "Western", "LA Clippers": "Western", "Los Angeles Lakers": "Western", "Phoenix Suns": "Western", "Sacramento Kings": "Western"
}

def calculate_team_performance():
    """Aggregate team performance per season from nba_games into team_performance."""
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql("SELECT * FROM nba_games", con)

    if df.empty:
        logging.info("No games found in nba_games table.")
        return

    performance_data = {}

    for _, game in df.iterrows():
        season = game["season"]
        home_team, away_team = game["home_team"], game["away_team"]
        home_score, away_score = game["home_score"], game["away_score"]
        winner = game["winner"]

        for team in [home_team, away_team]:
            if (team, season) not in performance_data:
                performance_data[(team, season)] = {
                    "wins": 0,
                    "losses": 0,
                    "points_scored": 0,
                    "points_allowed": 0,
                    "conference": CONFERENCE_MAP.get(team, "Unknown")
                }

        performance_data[(home_team, season)]["points_scored"] += home_score
        performance_data[(home_team, season)]["points_allowed"] += away_score
        performance_data[(away_team, season)]["points_scored"] += away_score
        performance_data[(away_team, season)]["points_allowed"] += home_score

        if winner == home_team:
            performance_data[(home_team, season)]["wins"] += 1
            performance_data[(away_team, season)]["losses"] += 1
        elif winner == away_team:
            performance_data[(away_team, season)]["wins"] += 1
            performance_data[(home_team, season)]["losses"] += 1

    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        for (team, season), stats in performance_data.items():
            wins, losses = stats["wins"], stats["losses"]
            points_scored, points_allowed = stats["points_scored"], stats["points_allowed"]
            win_percentage = wins / (wins + losses) if (wins + losses) > 0 else 0
            conference = stats["conference"]

            cur.execute("""
                INSERT INTO team_performance (team_name, conference, wins, losses, points_scored, points_allowed, win_percentage, season)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(team_name, season) DO UPDATE SET
                    conference=excluded.conference,
                    wins=excluded.wins,
                    losses=excluded.losses,
                    points_scored=excluded.points_scored,
                    points_allowed=excluded.points_allowed,
                    win_percentage=excluded.win_percentage
            """, (team, conference, wins, losses, points_scored, points_allowed, win_percentage, season))
        con.commit()

    logging.info("✔ Team performance calculation completed")

def load_team_performance(season=None):
    with sqlite3.connect(DB_PATH) as con:
        if season:
            query = "SELECT * FROM team_performance WHERE season=? ORDER BY win_percentage DESC"
            df = pd.read_sql(query, con, params=(season,))
        else:
            query = "SELECT * FROM team_performance ORDER BY season DESC, win_percentage DESC"
            df = pd.read_sql(query, con)
    return df