import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)

def fetch_nba_games(season=2025):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    logging.info(f"Fetching NBA games from {url}")

    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        logging.error(f"Failed to fetch games: {resp.status_code}")
        return pd.DataFrame()

    soup = BeautifulSoup(resp.content, "html.parser")
    table = soup.find("table", {"id": "schedule"})
    if table is None:
        logging.error("No table found")
        return pd.DataFrame()

    df = pd.read_html(str(table))[0]
    df = df.dropna(subset=["Date", "Visitor/Neutral", "PTS"])
    return df
