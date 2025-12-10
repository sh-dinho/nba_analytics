# ============================================================
# File: src/prediction_engine/predictor_cli.py
# Purpose: Command-line interface for NBAPredictor
# Project: nba_analysis
# Version: 1.5
# ============================================================

import click
import pandas as pd
from src.prediction_engine.predictor import NBAPredictor
from src.features.game_features import generate_features_for_games, fetch_season_games

@click.command()
@click.option("--model", required=True, help="Path to trained pipeline model (.pkl)")
@click.option("--limit", default=None, help="Limit number of games to fetch")
@click.option("--proba", is_flag=True, help="Output predicted probabilities")
@click.option("--label", is_flag=True, help="Output predicted labels")
def cli(model, limit, proba, label):
    """CLI for running NBA predictions."""
    game_ids = fetch_season_games(2025, limit=limit or 5)
    features = generate_features_for_games(game_ids)
    predictor = NBAPredictor(model_path=model)
    X = features.drop(columns=["win"], errors="ignore")
    output = pd.DataFrame(index=features.index)
    if proba:
        output["win_proba"] = predictor.predict_proba(X)
    if label:
        output["win_pred"] = predictor.predict_label(X)
    click.echo(output.to_json(orient="records"))

if __name__ == "__main__":
    cli()
