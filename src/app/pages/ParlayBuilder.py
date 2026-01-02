from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Page: Parlay Builder
# Purpose: Build and evaluate parlays using model or manual legs.
# ============================================================

import streamlit as st

from src.app.ui.header import render_header
from src.app.ui.navbar import render_navbar
from src.app.ui.floating_action_bar import render_floating_action_bar
from src.app.ui.page_state import set_active_page
from src.app.engines.parlay import ParlayLeg
from src.app.engines.betting_math import (
    american_to_decimal,
    parlay_decimal_odds,
    parlay_win_prob,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _get_parlay_legs() -> list[ParlayLeg]:
    return st.session_state.get("parlay_legs", [])


def _remove_leg(index: int) -> None:
    legs = _get_parlay_legs()
    if 0 <= index < len(legs):
        legs.pop(index)
        st.session_state["parlay_legs"] = legs


def _clear_parlay() -> None:
    st.session_state["parlay_legs"] = []


# ------------------------------------------------------------
# Main Page
# ------------------------------------------------------------
def main() -> None:
    set_active_page("Parlay Builder")

    render_header()
    render_navbar()

    st.title("🧩 Parlay Builder")

    legs = _get_parlay_legs()

    # --------------------------------------------------------
    # No legs yet
    # --------------------------------------------------------
    if not legs:
        st.info("No parlay legs added yet. Use the floating bar or this page to add legs.")
        render_floating_action_bar()
        return

    # --------------------------------------------------------
    # Display Current Legs
    # --------------------------------------------------------
    st.subheader("Current Parlay Legs")

    for i, leg in enumerate(legs, start=1):
        col1, col2 = st.columns([6, 1])
        with col1:
            st.write(
                f"**{i}. {leg.description}** — "
                f"odds `{leg.odds}` | win_prob `{leg.win_prob:.2f}`"
            )
        with col2:
            if st.button("❌ Remove", key=f"remove_leg_{i}"):
                _remove_leg(i - 1)
                st.experimental_rerun()

    # --------------------------------------------------------
    # Parlay Summary
    # --------------------------------------------------------
    st.subheader("Parlay Summary")

    decimal_odds = parlay_decimal_odds([american_to_decimal(l.odds) for l in legs])
    win_prob = parlay_win_prob([l.win_prob for l in legs])

    st.metric("Decimal Odds", f"{decimal_odds:.2f}")
    st.metric("Model Win Probability", f"{win_prob:.3f}")

    # Expected value (optional)
    ev = decimal_odds * win_prob - 1
    st.metric("Expected Value (EV per unit)", f"{ev:.3f}")

    # --------------------------------------------------------
    # Clear Parlay
    # --------------------------------------------------------
    if st.button("🗑️ Clear Parlay"):
        _clear_parlay()
        st.experimental_rerun()

    # --------------------------------------------------------
    # Floating Action Bar
    # --------------------------------------------------------
    render_floating_action_bar()


if __name__ == "__main__":
    main()
