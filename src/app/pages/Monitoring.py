from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Page: Monitoring
# Purpose: Display pipeline, ingestion, and data freshness health.
# ============================================================

import streamlit as st
from datetime import datetime, timedelta

from src.app.ui.header import render_header
from src.app.ui.navbar import render_navbar
from src.app.ui.floating_action_bar import render_floating_action_bar
from src.app.ui.page_state import set_active_page

from src.app.utils.pipeline_status import (
    get_pipeline_last_run,
    get_ingestion_last_run,
    time_since,
)

from src.config.paths import LONG_SNAPSHOT, FEATURES_SNAPSHOT


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _freshness_status(ts: datetime | None, threshold_days: int = 2) -> tuple[str, str]:
    """
    Return (emoji, message) based on freshness.
    """
    if ts is None:
        return "❌", "No timestamp found"

    delta = datetime.utcnow() - ts
    days = delta.days

    if days <= threshold_days:
        return "🟢", f"Fresh ({days} days old)"
    elif days <= threshold_days * 3:
        return "🟡", f"Stale ({days} days old)"
    else:
        return "🔴", f"Outdated ({days} days old)"


def _file_timestamp(path):
    if not path.exists():
        return None
    return datetime.utcfromtimestamp(path.stat().st_mtime)


# ------------------------------------------------------------
# Main Page
# ------------------------------------------------------------
def main() -> None:
    set_active_page("Monitoring")

    render_header()
    render_navbar()

    st.title("🩺 Pipeline & Ingestion Monitoring")

    # --------------------------------------------------------
    # Load timestamps
    # --------------------------------------------------------
    pipeline_ts = get_pipeline_last_run()
    ingestion_ts = get_ingestion_last_run()

    long_ts = _file_timestamp(LONG_SNAPSHOT)
    feat_ts = _file_timestamp(FEATURES_SNAPSHOT)

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------
    with col1:
        st.subheader("Pipeline")
        if pipeline_ts is None:
            st.error("No pipeline heartbeat found.")
        else:
            st.success(f"Last run: {pipeline_ts.isoformat()} (UTC)")
            st.caption(time_since(pipeline_ts))

    # --------------------------------------------------------
    # Ingestion
    # --------------------------------------------------------
    with col2:
        st.subheader("Ingestion")
        if ingestion_ts is None:
            st.error("No ingestion heartbeat found.")
        else:
            st.success(f"Last run: {ingestion_ts.isoformat()} (UTC)")
            st.caption(time_since(ingestion_ts))

    # --------------------------------------------------------
    # Data Freshness
    # --------------------------------------------------------
    with col3:
        st.subheader("Data Freshness")

        emoji_long, msg_long = _freshness_status(long_ts)
        emoji_feat, msg_feat = _freshness_status(feat_ts)

        st.write(f"**Long Snapshot:** {emoji_long} {msg_long}")
        st.write(f"**Feature Snapshot:** {emoji_feat} {msg_feat}")

    st.divider()

    # --------------------------------------------------------
    # System Summary
    # --------------------------------------------------------
    st.subheader("System Summary")

    ok_pipeline = pipeline_ts is not None
    ok_ingestion = ingestion_ts is not None
    ok_long = long_ts is not None
    ok_feat = feat_ts is not None

    overall_ok = all([ok_pipeline, ok_ingestion, ok_long, ok_feat])

    if overall_ok:
        st.success("System Status: All components healthy.")
    else:
        st.error("System Status: Issues detected.")

    render_floating_action_bar()


if __name__ == "__main__":
    main()
