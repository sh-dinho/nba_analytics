#!/usr/bin/env python
# ============================================================
# File: src/model_training/trainer_cli.py
# Purpose: Command-line interface for unified model training
# Project: nba_analysis
# Version: 2.0 (uses train_combined, adds model_type, SHAP, config)
# ============================================================

import click
import logging
from src.model_training.train_combined import train_model

@click.command()
@click.option("--data", required=True, help="Path to the data file for training (CSV or Parquet)")
@click.option("--output", default="models", help="Directory to save the trained model")
@click.option("--model_type", default="logreg", type=click.Choice(["logreg", "xgb"]), help="Model type to train")
@click.option("--run_shap", is_flag=True, help="Run SHAP analysis (XGB only)")
@click.option("--config", default="config.yaml", help="Path to config.yaml for MLflow settings")
def cli(data, output, model_type, run_shap, config):
    """CLI for running unified model training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        result = train_model(cache_file=data, out_dir=output,
                             model_type=model_type, run_shap=run_shap,
                             config_file=config)
        metrics = result["metrics"]
        click.echo(f"✅ {model_type.upper()} model trained. Saved to {result['model_path']}. "
                   f"Accuracy: {metrics.get('accuracy', 0):.3f}, LogLoss: {metrics.get('logloss', 0):.3f}")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        click.echo("❌ Training failed. See logs for details.")

if __name__ == "__main__":
    cli()
