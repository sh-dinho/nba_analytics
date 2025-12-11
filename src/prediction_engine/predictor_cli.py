#!/usr/bin/env python
# ============================================================
# File: src/prediction_engine/predictor_cli.py
# Purpose: Command-line interface for NBA Prediction
# Project: nba_analysis
# Version: 1.4 (named logger, safe error handling, clearer defaults)
# ============================================================

import click
import pandas as pd
import logging
import traceback

from src.prediction_engine.predictor import NBAPredictor
from src.features.feature_engineering import generate_features_for_games
from src.api.nba_api_client import fetch_season_games

logger = logging.getLogger("prediction_engine.predictor_cli")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@click.command()
@click.option("--model", required=True, help="Path to trained pipeline model (.pkl)")
@click.option("--season", default=2025, help="NBA season year")
@click.option("--limit", default=5, help="Limit number of games to fetch")
@click.option(
    "--proba",
    is_flag=True,
    help="Output predicted probabilities (default if no flag given)",
)
@click.option("--label", is_flag=True, help="Output predicted labels")
def cli(model, season, limit, proba, label):
    """CLI for running NBA predictions."""
    try:
        # Fetch games
        games = fetch_season_games(season, limit=limit)
        if games is None or games.empty:
            click.echo(f"❌ No games found for season {season}")
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

        logger.info("Generated predictions for %d games", len(output))
        click.echo(output.head().to_string(index=False))
        click.echo(output.to_json(orient="records", indent=2))

    except Exception as e:
        logger.error("Prediction run failed: %s", e)
        logger.debug(traceback.format_exc())
        click.echo("❌ Prediction run failed. See logs for details.")


if __name__ == "__main__":
    cli()
