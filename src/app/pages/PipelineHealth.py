from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Page: Pipeline Health Dashboard
# Purpose:
#     Unified operational view of:
#       • ingestion heartbeat
#       • pipeline heartbeat
#       • prediction file integrity
#       • canonical snapshot health
#       • backfill coverage
# ============================================================

import streamlit as st
from datetime import date, timedelta
from pathlib import Path
import pandas as pd

from src.app.ui.header import render_header
from src.app.ui.navbar import render_navbar
from src.app.ui.floating_action_bar import render_floating_action_bar
from src.app.ui.page_state import set_active_page

from src.app.utils.pipeline_status import (
    get_pipeline_last_run,
    get_ingestion_last_run,
    time_since,
)

from src.config.paths import DATA_DIR, LONG_SNAPSHOT
from src.app.ui.pipeline_controls import render_pipeline_controls

render_pipeline_controls()

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _prediction_file_status(pred_date: date) -> pd.DataFrame:
    """Check existence + size of prediction files."""
    base = DATA_DIR / "predictions"

    files = {
        "moneyline": base / f"moneyline_{pred_date}.parquet",
        "totals": base / f"totals_{pred_date}.parquet",
        "spread": base / f"spread_{pred_date}.parquet",
    }

    rows = []
    for label, path in files.items():
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        rows.append(
            {
                "market": label,
                "path": str(path),
                "exists": exists,
                "size_bytes": size,
                "status": "OK" if exists and size > 0 else "MISSING",
            }
        )

    return pd.DataFrame(rows)


def _snapshot_status() -> dict:
    """Check canonical long snapshot health."""
    if not LONG_SNAPSHOT.exists():
        return {
            "exists": False,
            "rows": 0,
            "path": str(LONG_SNAPSHOT),
            "status": "MISSING",
        }

    try:
        df = pd.read_parquet(LONG_SNAPSHOT)
        return {
            "exists": True,
            "rows": len(df),
            "path": str(LONG_SNAPSHOT),
            "status": "OK" if len(df) > 0 else "EMPTY",
        }
    except Exception:
        return {
            "exists": True,
            "rows": 0,
            "path": str(LONG_SNAPSHOT),
            "status": "CORRUPTED",
        }


# ------------------------------------------------------------
# Main Page
# ------------------------------------------------------------
def main() -> None:
    set_active_page("Pipeline Health")

    render_header()
    render_navbar()

    st.title("🩺 Pipeline Health Dashboard")

    # --------------------------------------------------------
    # Heartbeats
    # --------------------------------------------------------
    st.subheader("⏱ Heartbeats")

    pipeline_ts = get_pipeline_last_run()
    ingestion_ts = get_ingestion_last_run()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Pipeline")
        if pipeline_ts:
            st.success(f"Last run: {pipeline_ts.isoformat()}")
            st.caption(time_since(pipeline_ts))
        else:
            st.error("No pipeline heartbeat found.")

    with col2:
        st.markdown("### Ingestion")
        if ingestion_ts:
            st.success(f"Last run: {ingestion_ts.isoformat()}")
            st.caption(time_since(ingestion_ts))
        else:
            st.error("No ingestion heartbeat found.")

    st.divider()

    # --------------------------------------------------------
    # Prediction File Integrity
    # --------------------------------------------------------
    st.subheader("📦 Prediction File Integrity")

    pred_date = st.date_input("Prediction Date", value=date.today())
    df_pred = _prediction_file_status(pred_date)

    st.dataframe(df_pred, use_container_width=True)

    st.divider()

    # --------------------------------------------------------
    # Canonical Snapshot Health
    # --------------------------------------------------------
    st.subheader("📚 Canonical Snapshot")

    snap = _snapshot_status()

    if snap["status"] == "OK":
        st.success(f"Snapshot OK — {snap['rows']} rows")
    elif snap["status"] == "EMPTY":
        st.warning("Snapshot exists but is empty.")
    elif snap["status"] == "CORRUPTED":
        st.error("Snapshot file is corrupted.")
    else:
        st.error("Snapshot missing.")

    st.caption(f"Path: `{snap['path']}`")

    st.divider()

    # --------------------------------------------------------
    # Backfill Coverage
    # --------------------------------------------------------
    st.subheader("📅 Backfill Coverage (Last 14 Days)")

    base = DATA_DIR / "predictions"
    today = date.today()

    rows = []
    for i in range(14):
        d = today - timedelta(days=i)
        f = base / f"moneyline_{d}.parquet"
        rows.append(
            {
                "date": d,
                "exists": f.exists(),
                "status": "OK" if f.exists() else "MISSING",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    render_floating_action_bar()


if __name__ == "__main__":
    main()