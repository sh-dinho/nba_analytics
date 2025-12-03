# ============================================================
# File: scripts/dashboard.py
# Purpose: Generate a combined dashboard of feature trends and pipeline performance
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from core.config import RESULTS_DIR, FEATURES_LOG, PIPELINE_LOG
from core.log_config import setup_logger
from core.exceptions import PipelineError, DataError

logger = setup_logger("dashboard")


def main():
    """Generate a combined dashboard of feature trends and pipeline performance."""
    plt.figure(figsize=(14, 6))

    plotted = False

    # --- Plot Feature Trends ---
    if os.path.exists(FEATURES_LOG):
        try:
            features_df = pd.read_csv(FEATURES_LOG)
        except Exception as e:
            raise DataError(f"Failed to read features log: {e}")

        if not features_df.empty and {"timestamp", "avg_pts_diff", "avg_ast_diff", "avg_reb_diff"}.issubset(features_df.columns):
            plt.subplot(1, 2, 1)
            plt.plot(features_df["timestamp"], features_df["avg_pts_diff"], label="Avg PTS Diff", marker="o")
            plt.plot(features_df["timestamp"], features_df["avg_ast_diff"], label="Avg AST Diff", marker="o")
            plt.plot(features_df["timestamp"], features_df["avg_reb_diff"], label="Avg REB Diff", marker="o")
            plt.xlabel("Timestamp")
            plt.ylabel("Average Stat Difference")
            plt.title("Feature Trends Over Time")
            plt.legend()
            plt.xticks(rotation=45)
            plotted = True
        else:
            logger.warning("⚠️ Features log missing required columns or empty.")

    # --- Plot Pipeline Performance ---
    if os.path.exists(PIPELINE_LOG):
        try:
            pipeline_df = pd.read_csv(PIPELINE_LOG)
        except Exception as e:
            raise DataError(f"Failed to read pipeline log: {e}")

        if not pipeline_df.empty and {"timestamp", "final_bankroll_mean", "roi"}.issubset(pipeline_df.columns):
            plt.subplot(1, 2, 2)
            plt.plot(pipeline_df["timestamp"], pipeline_df["final_bankroll_mean"], label="Final Bankroll", marker="o")
            plt.plot(pipeline_df["timestamp"], pipeline_df["roi"] * 100, label="ROI (%)", marker="o")
            plt.xlabel("Timestamp")
            plt.ylabel("Value")
            plt.title("Pipeline Performance Trends")
            plt.legend()
            plt.xticks(rotation=45)
            plotted = True
        else:
            logger.warning("⚠️ Pipeline log missing required columns or empty.")

    if plotted:
        plt.tight_layout()
        chart_file = os.path.join(
            RESULTS_DIR,
            f"dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        try:
            plt.savefig(chart_file)
            logger.info(f"📊 Combined dashboard saved to {chart_file}")
        except Exception as e:
            raise PipelineError(f"Failed to save dashboard chart: {e}")
        finally:
            plt.close()
    else:
        logger.warning("⚠️ No data available to plot dashboard.")


if __name__ == "__main__":
    main()