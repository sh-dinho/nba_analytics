import logging

from scripts.scrape_and_load import load_season_to_db


def update_db():
    for season in [2023, 2024, 2025]:
        load_season_to_db(season)
    logging.info("✔ Database updated with latest seasons")