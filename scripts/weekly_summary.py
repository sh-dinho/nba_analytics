# File: scripts/weekly_summary.py
import pandas as pd
import os
import sys
import logging
import json
from datetime import datetime

# ----------------------------
# Logging setup
# ----------------------------
logger = logging.getLogger("weekly_summary")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def main(
    snapshots_file="results/weekly_snapshots.csv",
    out_file="results/weekly_summary.csv"
):
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(snapshots_file):
        raise FileNotFoundError("Run generate_weekly_snapshots.py first.")

    logger.info(f"📂 Loading snapshots from {snapshots_file}...")
    df = pd.read_csv(snapshots_file)

    # Validate required columns
    required_cols = {"team", "week"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"weekly_snapshots.csv missing required columns: {missing}")

    # Example aggregation: average stats per team per week
    summary = df.groupby(["team", "week"]).mean(numeric_only=True).reset_index()

    # Save summary
    summary.to_csv(out_file, index=False)
    ts_out_file = out_file.replace(".csv", f"_{_timestamp()}.csv")
    summary.to_csv(ts_out_file, index=False)

    logger.info(f"✅ Weekly summary saved to {out_file}")
    logger.info(f"📦 Timestamped backup saved to {ts_out_file}")
    logger.info(f"📊 Rows: {len(summary)}, Columns: {list(summary.columns)}")

    # Save metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "rows": len(summary),
        "columns": summary.columns.tolist(),
        "source_file": snapshots_file,
        "out_file": out_file
    }
    meta_file = "results/weekly_summary_meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"🧾 Metadata saved to {meta_file}")

    return summary


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Weekly summary generation failed: {e}")
        sys.exit(1)