from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
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
#       - dashboard.py (main dashboard)
#       - role-based tab visibility
# ============================================================

import time
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

SESSION_TIMEOUT_SECONDS = 60 * 60  # 1 hour


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def is_logged_in() -> bool:
    return "username" in st.session_state and st.session_state["username"] is not None


def is_admin() -> bool:
    return is_logged_in() and st.session_state.get("role") == "admin"


def logout():
    """Clear session state and return to login screen."""
    for key in ["username", "role", "client_id", "login_time"]:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()


def _session_expired() -> bool:
    """Check if the session has expired based on login_time."""
    if "login_time" not in st.session_state:
        return True
    return (time.time() - st.session_state["login_time"]) > SESSION_TIMEOUT_SECONDS


# ------------------------------------------------------------
# Login Gate
# ------------------------------------------------------------
def require_login():
    """
    Forces login before showing dashboard content.
    Handles:
      - login form
      - session timeout
      - role assignment
    """
    # If logged in but expired → logout
    if is_logged_in() and _session_expired():
        st.warning("Session expired. Please log in again.")
        logout()
        st.stop()

    # Already logged in → allow access
    if is_logged_in():
        return

    # Login form
    st.title("🔐 NBA Analytics v4 — Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username]["password"] == password:
            st.session_state["username"] = username
            st.session_state["role"] = USERS[username]["role"]
            st.session_state["client_id"] = USERS[username]["client_id"]
            st.session_state["login_time"] = time.time()
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

    st.stop()
