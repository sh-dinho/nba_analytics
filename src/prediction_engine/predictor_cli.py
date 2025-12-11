#!/usr/bin/env python
# ============================================================
# File: src/prediction_engine/predictor_cli.py
# Purpose: Command-line interface for NBA Prediction
# Project: nba_analysis
# Version: 1.3 (fixes imports, adds defaults and logging)
# ============================================================

import click
import pandas as pd
import logging

from src.prediction_engine.predictor import NBAPredictor
from src.features.feature_engineering import generate_features_for_games
from src.api.nba_api_client import fetch_season_games  # ✅ Added import

@click.command()
@click.option("--model", required=True, help="Path to trained pipeline model (.pkl)")
@click.option("--season", default=2025, help="NBA season year")
@click.option("--limit", default=5, help="Limit number of games to fetch")
@click.option("--proba", is_flag=True, help="Output predicted probabilities")
@click.option("--label", is_flag=True, help="Output predicted labels")
def cli(model, season, limit, proba, label):
    """CLI for running NBA predictions."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Fetch games
    games = fetch_season_games(season, limit=limit)
    if games.empty:
        click.echo("❌ No games found.")
        return

    # Generate features
    game_data_list = games.to_dict(orient="records")
    features = generate_features_for_games(game_data_list)

    predictor = NBAPredictor(model_path=model)
    X = features.drop(columns=["win"], errors="ignore")

    output = pd.DataFrame(index=features.index)
    if proba or (not proba and not label):  # default to proba
        output["win_proba"] = predictor.predict_proba(X)
    if label:
        output["win_pred"] = predictor.predict_label(X)

    # Show preview + JSON
    logging.info(f"Generated predictions for {len(output)} games")
    click.echo(output.head().to_string(index=False))
    click.echo(output.to_json(orient="records"))

if __name__ == "__main__":
    cli()
