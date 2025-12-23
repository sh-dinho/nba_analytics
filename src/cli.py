from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: CLI
# File: src/cli.py
# Author: Sadiq
#
# Description:
#     Thin command-line interface over the main pipelines.
# ============================================================

import argparse
from datetime import date

from src.config.logging import configure_logging
from src.pipeline.run_daily_predictions import run_daily_predictions
from src.ingestion.pipeline import run_today_ingestion, smart_ingest_season
from src.ingestion.health import run_ingestion_health_check
from src.pipeline.run_end_to_end import run_end_to_end
from src.model.training_core import train_and_register_model
from src.model.registry import promote_model
from src.config.paths import LONG_SNAPSHOT
import pandas as pd


def main():
    configure_logging()

    parser = argparse.ArgumentParser(prog="nba", description="NBA Analytics v4 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # --------------------------------------------------------
    # Daily predictions
    # --------------------------------------------------------
    daily = sub.add_parser("daily", help="Run daily predictions")
    daily.add_argument("--date", type=str, help="Prediction date (YYYY-MM-DD)")

    # --------------------------------------------------------
    # Ingestion
    # --------------------------------------------------------
    sub.add_parser("ingest-today", help="Ingest yesterday + today")
    season = sub.add_parser("ingest-season", help="Ingest full season")
    season.add_argument("year", type=int, help="Season start year (e.g., 2024)")

    # --------------------------------------------------------
    # Health check
    # --------------------------------------------------------
    sub.add_parser("ingestion-health", help="Run ingestion health checks")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    train = sub.add_parser("train", help="Train all v4 models")
    train.add_argument("--version", type=str, default="v4")

    # --------------------------------------------------------
    # End-to-end pipeline
    # --------------------------------------------------------
    sub.add_parser("run-e2e", help="Run full end-to-end pipeline")

    # --------------------------------------------------------
    # Promotion
    # --------------------------------------------------------
    promote = sub.add_parser("promote", help="Promote model to production")
    promote.add_argument("model_type", type=str, help="moneyline|totals|spread")
    promote.add_argument("--version", type=str, help="Specific version to promote")

    args = parser.parse_args()

    # --------------------------------------------------------
    # Command routing
    # --------------------------------------------------------

    if args.command == "daily":
        pred_date = date.fromisoformat(args.date) if args.date else date.today()
        run_daily_predictions(pred_date=pred_date)

    elif args.command == "ingest-today":
        run_today_ingestion(feature_version="v1")

    elif args.command == "ingest-season":
        smart_ingest_season(args.year, feature_version="v1")

    elif args.command == "ingestion-health":
        if not LONG_SNAPSHOT.exists():
            raise SystemExit("No canonical long snapshot found.")
        df = pd.read_parquet(LONG_SNAPSHOT)
        run_ingestion_health_check(df)

    elif args.command == "train":
        from src.features.builder import FeatureBuilder

        if not LONG_SNAPSHOT.exists():
            raise SystemExit("No canonical long snapshot found.")
        long_df = pd.read_parquet(LONG_SNAPSHOT)

        fb = FeatureBuilder(version=args.version)
        features_df = fb.build_from_long(long_df)

        for model_type in ["moneyline", "totals", "spread"]:
            train_and_register_model(
                model_type=model_type,
                df=features_df,
                feature_version=args.version,
            )

    elif args.command == "run-e2e":
        run_end_to_end()

    elif args.command == "promote":
        promote_model(args.model_type, version=args.version)


if __name__ == "__main__":
    main()
