from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Page: Backtest
# Purpose: Explore historical performance and bet logs.
# ============================================================

from pathlib import Path

import pandas as pd
import streamlit as st

from src.app.ui.header import render_header
from src.app.ui.navbar import render_navbar
from src.app.ui.floating_action_bar import render_floating_action_bar
from src.app.ui.page_state import set_active_page
from src.config.paths import DATA_DIR


def _load_bet_log() -> pd.DataFrame:
    path = DATA_DIR / "bets" / "bet_log.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def main() -> None:
    set_active_page("Backtest")

    render_header()
    render_navbar()

    st.title("📈 Backtest & Bet Log")

    df = _load_bet_log()

    if df.empty:
        st.warning("No bet log found yet. Place or import bets to see backtest results.")
        render_floating_action_bar()
        return

    st.success(f"Loaded {len(df)} bets from bet log.")

    if "pnl" in df.columns:
        st.subheader("Cumulative PnL")
        df_sorted = df.sort_values("date")
        df_sorted["cum_pnl"] = df_sorted["pnl"].cumsum()
        st.line_chart(df_sorted.set_index("date")["cum_pnl"])

    st.subheader("Raw Bet Log")
    st.dataframe(df, use_container_width=True)

    render_floating_action_bar()


if __name__ == "__main__":
    main()