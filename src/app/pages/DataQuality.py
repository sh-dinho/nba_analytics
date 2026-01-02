from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Page: Data Quality
# Purpose: Inspect canonical data health and schema sanity.
# ============================================================

from datetime import datetime
import pandas as pd
import streamlit as st

from src.app.ui.header import render_header
from src.app.ui.navbar import render_navbar
from src.app.ui.floating_action_bar import render_floating_action_bar
from src.app.ui.page_state import set_active_page
from src.config.paths import LONG_SNAPSHOT


# ------------------------------------------------------------
# Load Snapshot
# ------------------------------------------------------------
def _load_long_snapshot() -> pd.DataFrame:
    if not LONG_SNAPSHOT.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(LONG_SNAPSHOT)
    except Exception:
        return pd.DataFrame()


# ------------------------------------------------------------
# Main Page
# ------------------------------------------------------------
def main() -> None:
    set_active_page("Data Quality")

    render_header()
    render_navbar()

    st.title("🧪 Data Quality — Canonical Snapshot")

    df = _load_long_snapshot()

    # --------------------------------------------------------
    # Missing Snapshot
    # --------------------------------------------------------
    if df.empty:
        st.warning(
            "No canonical long snapshot found.\n"
            "Run ingestion + snapshot creation before using this page."
        )
        render_floating_action_bar()
        return

    st.success(f"Canonical snapshot loaded: {len(df):,} rows.")

    # --------------------------------------------------------
    # Freshness
    # --------------------------------------------------------
    st.subheader("Snapshot Freshness")

    try:
        ts = datetime.utcfromtimestamp(LONG_SNAPSHOT.stat().st_mtime)
        age_days = (datetime.utcnow() - ts).days
        freshness = (
            "🟢 Fresh" if age_days <= 1 else
            "🟡 Stale" if age_days <= 3 else
            "🔴 Outdated"
        )
        st.write(f"**Last updated:** {ts.isoformat()} UTC — {freshness} ({age_days} days old)")
    except Exception:
        st.error("Could not determine snapshot freshness.")

    st.divider()

    # --------------------------------------------------------
    # Schema Overview
    # --------------------------------------------------------
    st.subheader("Schema Overview")

    schema_df = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "non_null": df.notnull().sum().values,
        }
    )

    st.dataframe(schema_df, use_container_width=True)

    # --------------------------------------------------------
    # Missingness
    # --------------------------------------------------------
    st.subheader("Missingness Overview")

    missing = df.isnull().mean().sort_values(ascending=False)
    st.bar_chart(missing)

    # --------------------------------------------------------
    # Numeric Sanity Checks
    # --------------------------------------------------------
    st.subheader("Numeric Sanity Checks")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    issues = []

    for col in numeric_cols:
        max_val = df[col].abs().max()
        if pd.isna(max_val):
            issues.append(f"Column `{col}` contains only NaN values.")
        elif max_val > 1e9:
            issues.append(f"Column `{col}` has suspicious magnitude (max={max_val}).")

    if issues:
        for issue in issues:
            st.warning(issue)
    else:
        st.success("No numeric anomalies detected.")

    # --------------------------------------------------------
    # Latest Rows Preview
    # --------------------------------------------------------
    st.subheader("Latest 5 Rows")

    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            st.dataframe(df.sort_values("date").tail(5), use_container_width=True)
        except Exception:
            st.dataframe(df.tail(5), use_container_width=True)
    else:
        st.dataframe(df.tail(5), use_container_width=True)

    # --------------------------------------------------------
    # Floating Action Bar
    # --------------------------------------------------------
    render_floating_action_bar()


if __name__ == "__main__":
    main()
