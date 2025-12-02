# Updated load section
import pandas as pd
import os
import json
from datetime import datetime

os.makedirs("results", exist_ok=True)

def build_weekly_summary(notify=False, scale=True):
    features_file = "data/training_features.csv"
    if not os.path.exists(features_file):
        raise RuntimeError(f"{features_file} not found. Run fetch_player_stats_parallel_features.py first.")

    df = pd.read_csv(features_file)

    # Use existing weekly features
    summary = df.copy()

    # Optional scaling
    if scale:
        numeric_cols = summary.select_dtypes(include="number").columns.difference(["playerid", "week", "year"])
        for col in numeric_cols:
            mean, std = summary[col].mean(), summary[col].std(ddof=0)
            summary[f"{col}_zscaled"] = (summary[col] - mean) / (std or 1)

    # Save weekly summary
    out_file = "results/weekly_summary.csv"
    summary.to_csv(out_file, index=False)

    meta = {
        "generated_at": datetime.now().isoformat(),
        "rows": len(summary),
        "columns": summary.columns.tolist(),
        "source_file": features_file,
        "scaled": scale
    }
    meta_file = "results/weekly_summary_meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    if notify:
        print(f"✅ Weekly summary built and saved to {out_file}")
        print(f"📦 Metadata saved to {meta_file}")
        print(f"📊 Rows: {len(summary)}, Columns: {len(summary.columns)}")

    return summary
