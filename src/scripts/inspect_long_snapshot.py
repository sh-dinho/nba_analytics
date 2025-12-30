from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.config.paths import LONG_SNAPSHOT


def inspect_long_snapshot():
    print("=== Inspecting LONG_SNAPSHOT ===")
    print(f"Path: {LONG_SNAPSHOT}")

    if not LONG_SNAPSHOT.exists():
        print("ERROR: LONG_SNAPSHOT does not exist.")
        return

    df = pd.read_parquet(LONG_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"])

    print("\n--- Columns ---")
    print(df.columns.tolist())

    # Identify identity-like columns
    identity_candidates = []
    for col in df.columns:
        if (
            "id" in col.lower()
            or "team" in col.lower()
            or "opponent" in col.lower()
            or "date" in col.lower()
            or "home" in col.lower()
            or "away" in col.lower()
            or "status" in col.lower()
            or "schema" in col.lower()
        ):
            identity_candidates.append(col)

    print("\n--- Identity-like columns (candidates) ---")
    print(identity_candidates)

    # Numeric vs non-numeric
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    print("\n--- Numeric columns ---")
    print(numeric_cols)

    print("\n--- Non-numeric columns ---")
    print(non_numeric_cols)

    # Columns with NaNs
    nan_cols = df.columns[df.isna().any()].tolist()
    print("\n--- Columns containing NaN values ---")
    print(nan_cols)

    # Summary
    print("\n=== Summary ===")
    print(f"Total columns: {len(df.columns)}")
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Non-numeric columns: {len(non_numeric_cols)}")
    print(f"Identity-like columns: {len(identity_candidates)}")

    print("\nRecommended identity columns to drop before prediction:")
    print(identity_candidates)


if __name__ == "__main__":
    inspect_long_snapshot()
