# File: scripts/dashboard.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scripts.utils import setup_logger

logger = setup_logger("dashboard")

RESULTS_DIR = "results"
FEATURES_LOG = os.path.join(RESULTS_DIR, "features_summary.csv")
PIPELINE_LOG = os.path.join(RESULTS_DIR, "pipeline_metrics.csv")

def main():
    """Generate a combined dashboard of feature trends and pipeline performance."""
    plt.figure(figsize=(14, 6))

    # --- Plot Feature Trends ---
    if os.path.exists(FEATURES_LOG):
        features_df = pd.read_csv(FEATURES_LOG)
        if not features_df.empty:
            plt.subplot(1, 2, 1)
            plt.plot(features_df["timestamp"], features_df["avg_pts_diff"], label="Avg PTS Diff", marker="o")
            plt.plot(features_df["timestamp"], features_df["avg_ast_diff"], label="Avg AST Diff", marker="o")
            plt.plot(features_df["timestamp"], features_df["avg_reb_diff"], label="Avg REB Diff", marker="o")
            plt.xlabel("Timestamp")
            plt.ylabel("Average Stat Difference")
            plt.title("Feature Trends Over Time")
            plt.legend()
            plt.xticks(rotation=45)

    # --- Plot Pipeline Performance ---
    if os.path.exists(PIPELINE_LOG):
        pipeline_df = pd.read_csv(PIPELINE_LOG)
        if not pipeline_df.empty:
            plt.subplot(1, 2, 2)
            plt.plot(pipeline_df["timestamp"], pipeline_df["final_bankroll_mean"], label="Final Bankroll", marker="o")
            plt.plot(pipeline_df["timestamp"], pipeline_df["roi"] * 100, label="ROI (%)", marker="o")
            plt.xlabel("Timestamp")
            plt.ylabel("Value")
            plt.title("Pipeline Performance Trends")
            plt.legend()
            plt.xticks(rotation=45)

    plt.tight_layout()
    chart_file = os.path.join(
        RESULTS_DIR,
        f"dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    plt.savefig(chart_file)
    logger.info(f"📊 Combined dashboard saved to {chart_file}")

if __name__ == "__main__":
    main()