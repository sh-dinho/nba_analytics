# ============================================================
# 🏀 NBA Analytics v3
# Module: Dashboard Authentication
# File: src/dashboard/auth.py
# Author: Sadiq
#
# Description:
#     Lightweight session-based authentication for the Streamlit
#     dashboard. Supports:
#       - admin role (full access)
#       - client role (restricted access)
#
#     Integrates with:
#       - app.py (main dashboard)
#       - role-based tab visibility
# ============================================================

from __future__ import annotations

import streamlit as st

# ------------------------------------------------------------
# User database (replace with secure store in production)
# ------------------------------------------------------------
USERS = {
    "admin": {
        "password": "admin123",
        "role": "admin",
        "client_id": None,
    },
    "client": {
        "password": "client123",
        "role": "client",
        "client_id": "CLIENT_A",
    },
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def is_logged_in() -> bool:
    return "username" in st.session_state and st.session_state["username"] is not None


def is_admin() -> bool:
    return is_logged_in() and st.session_state.get("role") == "admin"


def require_login():
    """
    Forces login before showing dashboard content.
    """
    if is_logged_in():
        return

    st.title("🔐 NBA Analytics v3 — Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username]["password"] == password:
            st.session_state["username"] = username
            st.session_state["role"] = USERS[username]["role"]
            st.session_state["client_id"] = USERS[username]["client_id"]
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

    st.stop()
