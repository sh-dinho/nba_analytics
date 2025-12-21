import streamlit as st

USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "client": {"password": "client123", "role": "client"},
}


def login():
    st.title("NBA Analytics Client Portal")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username]["password"] == password:
            st.session_state["authenticated"] = True
            st.session_state["role"] = USERS[username]["role"]
            st.session_state["username"] = username
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")


def require_login():
    if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
        login()
        st.stop()


def is_admin():
    return st.session_state.get("role") == "admin"
