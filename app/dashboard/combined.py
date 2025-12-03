# ============================================================
# File: app/dashboard/combined.py
# Purpose: Generate a master dashboard overlaying daily, weekly, monthly trends
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from core.config import RESULTS_DIR
from core.log_config import setup_logger
from core.exceptions import PipelineError, DataError
from scripts.notifications import send_photo

logger = setup_logger("combined_dashboard")

def main():
    daily_file = os.path.join(RESULTS_DIR, "summary.csv")
    weekly_file = os.path.join(RESULTS_DIR, "weekly_summary.csv")
    monthly_file = os.path.join(RESULTS_DIR, "monthly_summary.csv")

    dfs = {}
    for name, path in [("Daily", daily_file), ("Weekly", weekly_file), ("Monthly", monthly_file)]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if not df.empty and "Final_Bankroll" in df.columns:
                    dfs[name] = df
            except Exception as e:
                raise DataError(f"Failed to read {name} summary: {e}")

    if not dfs:
        logger.warning("⚠️ No summary files available for combined dashboard.")
        return

    plt.figure(figsize=(12, 6))
    for name, df in dfs.items():
        x = df["Date"] if "Date" in df.columns else df.index
        plt.plot(x, df["Final_Bankroll"], marker="o", label=f"{name} Bankroll")

    plt.xlabel("Time")
    plt.ylabel("Final Bankroll")
    plt.title("Combined Bankroll Trajectories")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    chart_file = os.path.join(RESULTS_DIR, "combined_dashboard.png")
    try:
        plt.savefig(chart_file)
        logger.info(f"📊 Combined dashboard saved to {chart_file}")
        send_photo(chart_file, caption="📊 Combined Dashboard")
    except Exception as e:
        raise PipelineError(f"Failed to save combined dashboard chart: {e}")
    finally:
        plt.close()

if __name__ == "__main__":
    main()