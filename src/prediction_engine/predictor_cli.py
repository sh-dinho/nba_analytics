# ============================================================
# Path: src/prediction_engine/predictor_cli.py
# Filename: predictor_cli.py
# Author: Your Team
# Date: December 2025
# Purpose: Command-line interface for NBAPredictor with date filtering
# ============================================================

import click
import pandas as pd
from src.prediction_engine.predictor import NBAPredictor
from features.game_features import generate_features_for_games, fetch_season_games
from nba_api.stats.endpoints import leaguegamefinder

def fetch_games_by_date(year: int, game_date: str) -> list[str]:
    """Fetch NBA game IDs for a specific date (YYYY-MM-DD)."""
    finder = leaguegamefinder.LeagueGameFinder(season_nullable=str(year))
    df = finder.get_data_frames()[0]
    df_today = df[df["GAME_DATE"] == game_date]
    return df_today["GAME_ID"].tolist()

@click.command()
@click.option("--limit", default=None, help="Limit number of games to fetch.")
@click.option("--date", default=None, help="Filter games by date (YYYY-MM-DD).")
@click.option("--proba", is_flag=True, help="Output predicted probabilities.")
@click.option("--label", is_flag=True, help="Output predicted labels.")
@click.option("--tags", is_flag=True, help="Output predictions with tags.")
@click.option("--all", is_flag=True, help="Output both probabilities and labels together.")
def cli(limit, date, proba, label, tags, all):
    """
    CLI for running NBA predictions.
    Fetches games by date or limit, generates features, loads model, and outputs predictions.
    """
    # -----------------------------
    # Step 1: Fetch games
    # -----------------------------
    if date:
        game_ids = fetch_games_by_date(2025, date)
    else:
        game_ids = fetch_season_games(2025, limit=limit or 5)

    features = generate_features_for_games(game_ids)

    # -----------------------------
    # Step 2: Load predictor
    # -----------------------------
    predictor = NBAPredictor()
    X = features.drop(columns=["win"])
    y = features["win"]

    # Fit model if not already fitted
    if not hasattr(predictor.model, "classes_"):
        predictor.model.fit(X, y)

    # -----------------------------
    # Step 3: Make predictions
    # -----------------------------
    output = pd.DataFrame(index=features.index)

    if all:
        output["proba"] = predictor.predict_proba(X)
        output["label"] = predictor.predict(X).tolist()
    else:
        if proba:
            output["proba"] = predictor.predict_proba(X)
        if label:
            output["label"] = predictor.predict(X).tolist()
        if tags:
            output["tags"] = [["NBA", "prediction"] for _ in range(len(output))]

    # -----------------------------
    # Step 4: Print JSON output
    # -----------------------------
    click.echo(output.to_json(orient="records"))
