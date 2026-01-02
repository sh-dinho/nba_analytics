from __future__ import annotations

# ============================================================
# NBA Analytics Engine — Ingestion Runner (Canonical)
# File: src/scripts/run_ingestion.py
# Author: Sadiq
# ============================================================

import sys
from loguru import logger

from src.config.config_validator import validate_config, print_config_report
from src.ingestion.orchestrator import run_full_ingestion


def main() -> None:
    logger.info("🏀 Starting NBA Analytics ingestion pipeline...")

    # --------------------------------------------------------
    # 1. Validate configuration
    # --------------------------------------------------------
    try:
        report = validate_config(auto_create_dirs=True)
        print_config_report(report)
    except Exception as e:
        logger.exception(f"❌ Configuration validation crashed: {e}")
        sys.exit(1)

    if not report.get("ok", False):
        logger.error("❌ Configuration validation failed. Aborting ingestion.")
        sys.exit(1)

    # --------------------------------------------------------
    # 2. Run ingestion
    # --------------------------------------------------------
    try:
        logger.info("🚀 Running ingestion orchestrator...")
        df = run_full_ingestion()

        if df is None:
            logger.error("❌ Ingestion returned None (unexpected).")
            sys.exit(1)

        if df.empty:
            logger.error("❌ Ingestion completed but NO data was collected.")
            sys.exit(1)

        logger.success(f"✅ Ingestion successful. Processed {len(df)} rows.")

    except Exception as e:
        logger.exception(f"❌ Ingestion pipeline failed: {e}")
        sys.exit(1)

    # --------------------------------------------------------
    # 3. Final success banner
    # --------------------------------------------------------
    logger.info("🎉 Ingestion pipeline completed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
