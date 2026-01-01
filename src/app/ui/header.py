from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Streamlit UI Header
# File: src/app/ui/header.py
# Purpose: Render global app header with version and tagline.
# ============================================================

import streamlit as st

APP_VERSION = "5.0.0"


def render_header() -> None:
    st.markdown(
        f"""
        <div style="
            padding: 12px 0;
            border-bottom: 1px solid #444;
            margin-bottom: 10px;
        ">
            <h1 style="margin: 0; font-size: 30px;">
                🏀 NBA Analytics v{APP_VERSION}
            </h1>
            <p style="margin: 0; font-size: 14px; color: #aaa;">
                Automated recommendations • Smart parlays • Model registry • Pipeline health • Bet tracking • Simulation lab
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
