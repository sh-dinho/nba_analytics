# ============================================================
# File: app/dashboard/dashboard_runner.py
# Purpose: Run all dashboards (daily, weekly, monthly, combined) in sequence
# ============================================================

import logging
from app.dashboard import daily, weekly, monthly, combined

logger = logging.getLogger("dashboard_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logger.info("🚀 Starting dashboard generation sequence")

    try:
        logger.info("▶️ Generating Daily Dashboard...")
        daily.main()
    except Exception as e:
        logger.error(f"❌ Daily dashboard failed: {e}")

    try:
        logger.info("▶️ Generating Weekly Dashboard...")
        weekly.main()
    except Exception as e:
        logger.error(f"❌ Weekly dashboard failed: {e}")

    try:
        logger.info("▶️ Generating Monthly Dashboard...")
        monthly.main()
    except Exception as e:
        logger.error(f"❌ Monthly dashboard failed: {e}")

    try:
        logger.info("▶️ Generating Combined Dashboard...")
        combined.main()
    except Exception as e:
        logger.error(f"❌ Combined dashboard failed: {e}")

    logger.info("✅ Dashboard generation sequence complete")

if __name__ == "__main__":
    main()