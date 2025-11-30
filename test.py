# import yaml
# import sys

# CONFIG_PATH = "config.yaml"

# try:
#     with open(CONFIG_PATH) as f:
#         CONFIG = yaml.safe_load(f)
# except yaml.YAMLError as e:
#     print(f"❌ YAML parsing error: {e}")
#     sys.exit(1)
# import requests

# season = 2024
# page = 1
# url = f"https://www.balldontlie.io/api/v1/games?seasons[]={season}&page={page}"
# resp = requests.get(url)
# resp.raise_for_status()
# data = resp.json()
import sqlite3
import pandas as pd

conn = sqlite3.connect("./nba_analytics.db")
df = pd.read_sql("SELECT * FROM nba_games LIMIT 5", conn)
conn.close()
print(df.head())
