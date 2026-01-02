from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Page: Parlay Analytics
# Purpose: Analyze performance of parlays as a distinct product.
# ============================================================

import streamlit as st

from src.app.ui.header import render_header
from src.app.ui.navbar import render_navbar
from src.app.ui.floating_action_bar import render_floating_action_bar
from src.app.ui.page_state import set_active_page

from src.app.utils.bets_io import load_bet_log
from src.app.engines.bet_analytics import aggregate_parlays


def main() -> None:
    set_active_page("Parlay Analytics")

    render_header()
    render_navbar()

    st.title("🧩 Parlay Analytics")

    df = load_bet_log()

    if df.empty:
        st.warning("No bets logged yet.")
        render_floating_action_bar()
        return

    if "parlay_group_id" not in df.columns:
        st.warning("No 'parlay_group_id' column found. Parlays are not tracked as groups.")
        render_floating_action_bar()
        return

    parlays = aggregate_parlays(df)
    if parlays.empty:
        st.info("No parlay data available.")
        render_floating_action_bar()
        return

    st.success(f"Loaded {len(parlays):,} parlays.")

    # Summary metrics
    st.subheader("Parlay Summary")

    total_parlays = len(parlays)
    total_stake = float(parlays["stake"].sum())
    total_pnl = float(parlays["pnl"].sum())
    avg_legs = float(parlays["legs"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Parlays", total_parlays)
    c2.metric("Total Stake", f"{total_stake:.2f}")
    c3.metric("Total PnL", f"{total_pnl:.2f}")
    c4.metric("Avg Legs per Parlay", f"{avg_legs:.2f}")

    if "roi" in parlays.columns:
        avg_roi = float(parlays["roi"].mean())
        st.metric("Average Parlay ROI", f"{avg_roi:.2%}")

    st.divider()

    # Table
    st.subheader("Parlay Log")
    st.dataframe(parlays, use_container_width=True)

    # ROI vs legs count
    st.subheader("ROI by Number of Legs")
    by_legs = (
        parlays.groupby("legs")
        .agg(
            parlays=("parlay_group_id", "count"),
            stake=("stake", "sum"),
            pnl=("pnl", "sum"),
            roi=("roi", "mean"),
        )
        .reset_index()
    )

    st.dataframe(by_legs, use_container_width=True)
    if "roi" in by_legs.columns:
        st.bar_chart(by_legs.set_index("legs")["roi"])

    render_floating_action_bar()


if __name__ == "__main__":
    main()
