from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Page: Bet Tracker
# Purpose: View and analyze logged bets.
# ============================================================

import pandas as pd
import streamlit as st

from src.app.ui.header import render_header
from src.app.ui.navbar import render_navbar
from src.app.ui.floating_action_bar import render_floating_action_bar
from src.app.ui.page_state import set_active_page
from src.config.paths import DATA_DIR


# ------------------------------------------------------------
# Load Bet Log
# ------------------------------------------------------------
def _load_bet_log() -> pd.DataFrame:
    path = DATA_DIR / "bets" / "bet_log.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


# ------------------------------------------------------------
# Main Page
# ------------------------------------------------------------
def main() -> None:
    set_active_page("Bet Tracker")

    render_header()
    render_navbar()

    st.title("📒 Bet Tracker")

    df = _load_bet_log()

    # --------------------------------------------------------
    # No bets yet
    # --------------------------------------------------------
    if df.empty:
        st.warning("No bets logged yet. Use the floating action bar to log bets.")
        render_floating_action_bar()
        return

    st.success(f"Loaded {len(df):,} bets.")

    # Ensure date is parsed
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------
    with st.expander("Filters", expanded=False):
        if "date" in df.columns:
            unique_dates = sorted(df["date"].dropna().dt.date.unique())
            selected_dates = st.multiselect("Filter by date", unique_dates, default=unique_dates)
            df = df[df["date"].dt.date.isin(selected_dates)]

        if "team" in df.columns:
            teams = sorted(df["team"].dropna().unique())
            selected_teams = st.multiselect("Filter by team", teams)
            if selected_teams:
                df = df[df["team"].isin(selected_teams)]

        if "result" in df.columns:
            results = sorted(df["result"].dropna().unique())
            selected_results = st.multiselect("Filter by result", results)
            if selected_results:
                df = df[df["result"].isin(selected_results)]

    # --------------------------------------------------------
    # Summary Metrics
    # --------------------------------------------------------
    st.subheader("Summary")

    total_bets = len(df)
    total_stake = df["stake"].sum() if "stake" in df.columns else 0
    total_pnl = df["pnl"].sum() if "pnl" in df.columns else 0

    win_rate = None
    if "result" in df.columns:
        wins = (df["result"] == "win").sum()
        win_rate = wins / total_bets if total_bets > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Bets", total_bets)
    col2.metric("Total Stake", f"{total_stake:.2f}")
    col3.metric("Total PnL", f"{total_pnl:.2f}")
    if win_rate is not None:
        col4.metric("Win Rate", f"{win_rate:.1%}")

    st.divider()

    # --------------------------------------------------------
    # Bet Log Table
    # --------------------------------------------------------
    st.subheader("Bet Log")
    st.dataframe(df, use_container_width=True)

    # --------------------------------------------------------
    # PnL Over Time
    # --------------------------------------------------------
    if "pnl" in df.columns and "date" in df.columns:
        st.subheader("PnL Over Time")

        df_sorted = df.sort_values("date")
        df_sorted["cum_pnl"] = df_sorted["pnl"].cumsum()

        st.line_chart(df_sorted.set_index("date")["cum_pnl"])

    # --------------------------------------------------------
    # Market Breakdown
    # --------------------------------------------------------
    if "market" in df.columns:
        st.subheader("Market Breakdown")
        st.bar_chart(df["market"].value_counts())

    # --------------------------------------------------------
    # Floating Action Bar
    # --------------------------------------------------------
    render_floating_action_bar()


if __name__ == "__main__":
    main()
