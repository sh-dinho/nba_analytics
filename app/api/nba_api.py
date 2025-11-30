import requests
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

NBA_API_URL = "https://www.balldontlie.io/api/v1/games"
PER_PAGE = 100

def fetch_nba_games(season=2025):
    logging.info(f"Fetching NBA games data for season {season}...")
    games = []
    page = 1

    while True:
        url = f"{NBA_API_URL}?seasons[]={season}&per_page={PER_PAGE}&page={page}"
        response = requests.get(url)

        if response.status_code != 200:
            logging.warning(f"Error fetching data for page {page}: {response.status_code}")
            break
        
        data = response.json()
        games.extend(data['data'])

        if page >= data['meta']['total_pages']:
            break
        page += 1

    if not games:
        logging.error("No NBA game data found!")
        return pd.DataFrame()

    return pd.DataFrame(games)

def clean_data(df):
    """Clean and preprocess the data."""
    df['date'] = pd.to_datetime(df['date'])
    df['home_team'] = df['home_team'].apply(lambda x: x['full_name'])
    df['visitor_team'] = df['visitor_team'].apply(lambda x: x['full_name'])
    df['home_score'] = df['home_team_score']
    df['away_score'] = df['visitor_team_score']
    df['home_win'] = df.apply(lambda row: 1 if row['home_score'] > row['away_score'] else 0, axis=1)
    return df[['date', 'home_team', 'visitor_team', 'home_score', 'away_score', 'home_win']]
