from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Floating Action Bar
# File: src/app/ui/floating_action_bar.py
# Purpose: Global floating bar with quick actions:
#          log bet, add parlay leg, refresh predictions.
# ============================================================

from datetime import date

import streamlit as st

from src.app.engines.bet_tracker import BetRecord, append_bet, new_bet_id
from src.app.engines.parlay import ParlayLeg


def render_floating_action_bar() -> None:
    """
    Global floating action bar with:
      - Log Bet
      - Add to Parlay
      - Refresh Predictions
    """

    st.markdown(
        """
        <style>
            .floating-bar {
                position: fixed;
                bottom: 25px;
                right: 25px;
                background: rgba(30, 30, 30, 0.95);
                padding: 14px 18px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
                z-index: 9999;
                display: flex;
                gap: 14px;
            }
            .fab-btn {
                background-color: #2ecc71;
                color: black;
                padding: 8px 14px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: bold;
                font-size: 14px;
            }
            .fab-btn:hover {
                background-color: #27ae60;
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Container
    st.markdown('<div class="floating-bar">', unsafe_allow_html=True)

    if st.button("📝 Log Bet", key="fab_log_bet"):
        st.session_state["show_log_bet_modal"] = True

    if st.button("➕ Add to Parlay", key="fab_add_parlay"):
        st.session_state["show_add_parlay_modal"] = True

    if st.button("🔄 Refresh", key="fab_refresh"):
        st.session_state["refresh_predictions"] = True
        st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Log Bet Modal
    # -----------------------------
    if st.session_state.get("show_log_bet_modal", False):
        with st.modal("Log a Bet"):
            team = st.text_input("Team")
            opponent = st.text_input("Opponent")
            odds = st.number_input("Odds (American)", value=-110.0)
            stake = st.number_input("Stake", value=100.0)
            desc = st.text_input("Bet Description", value="Manual Bet")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit Bet"):
                    if not team or not opponent:
                        st.error("Team and opponent are required.")
                    else:
                        record = BetRecord(
                            bet_id=new_bet_id(),
                            date=str(date.today()),
                            game_date=str(date.today()),
                            market="manual",
                            team=team,
                            opponent=opponent,
                            bet_description=desc,
                            odds=float(odds),
                            stake=float(stake),
                            result="pending",
                            payout=0.0,
                            source="manual",
                        )
                            # confidence fields left None for manual entries
                        append_bet(record)
                        st.success(f"Bet logged. Bet ID: {record.bet_id}")
                        st.session_state["show_log_bet_modal"] = False

            with col2:
                if st.button("Close"):
                    st.session_state["show_log_bet_modal"] = False

    # -----------------------------
    # Add Parlay Leg Modal
    # -----------------------------
    if st.session_state.get("show_add_parlay_modal", False):
        with st.modal("Add Parlay Leg"):
            desc = st.text_input("Leg Description")
            odds = st.number_input("Odds (American)", value=-110.0)
            win_prob = st.number_input(
                "Win Probability (0-1)",
                value=0.55,
                min_value=0.0,
                max_value=1.0,
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Add Leg"):
                    if "parlay_legs" not in st.session_state:
                        st.session_state["parlay_legs"] = []
                    st.session_state["parlay_legs"].append(
                        ParlayLeg(
                            description=desc,
                            odds=float(odds),
                            win_prob=float(win_prob),
                            source="manual",
                        )
                    )
                    st.success("Leg added to parlay.")
                    st.session_state["show_add_parlay_modal"] = False

            with col2:
                if st.button("Close"):
                    st.session_state["show_add_parlay_modal"] = False
