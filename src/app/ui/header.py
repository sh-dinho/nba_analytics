from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Streamlit UI Header
# File: src/app/ui/header.py
# Purpose:
#     Render global app header with:
#       • version badge
#       • tagline
#       • consistent styling across pages
# ============================================================

import streamlit as st

APP_VERSION = "5.0.0"


def render_header() -> None:
    """Render the global application header."""

    # Global style block (keeps things consistent across pages)
    st.markdown(
        """
        <style>
            .nba-header {
                padding: 12px 0 6px 0;
                border-bottom: 1px solid #444;
                margin-bottom: 14px;
            }
            .nba-title {
                margin: 0;
                font-size: 30px;
                font-weight: 700;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .nba-version-badge {
                background: #2ecc71;
                color: black;
                padding: 2px 8px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }
            .nba-tagline {
                margin: 0;
                margin-top: 4px;
                font-size: 14px;
                color: #aaa;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header block
    st.markdown(
        f"""
        <div class="nba-header">
            <div class="nba-title">
                🏀 NBA Analytics
                <span class="nba-version-badge">v{APP_VERSION}</span>
            </div>
            <p class="nba-tagline">
                Automated recommendations • Smart parlays • Model registry • Pipeline health • Bet tracking • Simulation lab
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )