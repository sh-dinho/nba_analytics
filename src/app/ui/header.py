from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Streamlit UI Header
# File: src/app/ui/header.py
# ============================================================

import streamlit as st


def render_header() -> None:
    st.markdown(
        """
        <div style="padding: 10px 0; border-bottom: 1px solid #444;">
            <h1 style="margin: 0; font-size: 30px;">🏀 NBA Analytics v4</h1>
            <p style="margin: 0; font-size: 14px; color: #888;">
                Canonical ingestion • v4 features • schema-aware models • auto-promotion • betting analytics
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
