# File: scripts/run_daily_pipeline_cli.py

import argparse
from scripts.utils import setup_logger
from app.prediction_pipeline import PredictionPipeline

logger = setup_logger("NBA_CLI")

def main():
    parser = argparse.ArgumentParser(description="NBA Analytics CLI")
    parser.add_argument("--threshold", type=float, default=0.6, help="Probability threshold for bets")
    parser.add_argument("--strategy", choices=["kelly", "flat"], default="kelly", help="Bet sizing strategy")
    parser.add_argument("--max_fraction", type=float, default=0.05, help="Max fraction of bankroll per bet")
    # --- NEW ARGUMENT ---
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic player data instead of API fetch")
    # --------------------
    args = parser.parse_args()

    # --- UPDATE logger.info ---
    logger.info(f"Starting pipeline | threshold={args.threshold}, strategy={args.strategy}, max_fraction={args.max_fraction}, use_synthetic={args.use_synthetic}")

    pipeline = PredictionPipeline(
        threshold=args.threshold,
        strategy=args.strategy,
        max_fraction=args.max_fraction,
        # --- PASS NEW ARGUMENT ---
        use_synthetic=args.use_synthetic
        # -------------------------
    )

    try:
        df, metrics = pipeline.run()
        if metrics:
            logger.info("✅ CLI run completed successfully")
            logger.info(f"Games processed: {len(df)}")
            logger.info(f"Final bankroll: {metrics.get('final_bankroll_mean', 0):.2f}")
            logger.info(f"ROI: {metrics.get('roi', 0)*100:.2f}%")
        else:
            logger.warning("Pipeline completed but no metrics were returned.")
    except Exception as e:
        logger.error(f"❌ CLI run failed: {e}")
        raise

if __name__ == "__main__":
    main()