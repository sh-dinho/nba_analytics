from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Game Breakdown
# File: src/app/pages/09_Game_Breakdown.py
# Purpose: Per-game view showing moneyline, total, and spread
#          edges, with direct parlay and bet logging actions.
# ============================================================

from datetime import date

import pandas as pd
import streamlit as st

from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar
from src.model.predict import run_prediction_for_date
from src.model.predict_totals import run_totals_prediction_for_date
from src.model.predict_spread import run_spread_prediction_for_date
from src.app.engines.parlay import ParlayLeg
from src.app.engines.bet_tracker import BetRecord, append_bet, new_bet_id

st.set_page_config(page_title="Game Breakdown", page_icon="📊", layout="wide")

render_header()
set_active_page("Game Breakdown")
render_navbar()

st.title("📊 Game Breakdown")

today = date.today()
pred_date = st.date_input("Prediction date", value=today)

ml = run_prediction_for_date(pred_date)
tot = run_totals_prediction_for_date(pred_date)
sp = run_spread_prediction_for_date(pred_date)

if ml.empty:
    st.info("No predictions available for this date.")
    st.stop()

games = ml[["game_id", "home_team", "away_team"]].drop_duplicates()

selected_game = st.selectbox(
    "Select a game",
    options=games["game_id"],
    format_func=lambda gid: f"{games.loc[games['game_id']==gid, 'away_team'].iloc[0]} @ {games.loc[games['game_id']==gid, 'home_team'].iloc[0]}",
)

game_ml = ml[ml["game_id"] == selected_game]
game_tot = tot[tot["game_id"] == selected_game] if not tot.empty else pd.DataFrame()
game_sp = sp[sp["game_id"] == selected_game] if not sp.empty else pd.DataFrame()

st.markdown("### Moneyline")

st.dataframe(
    game_ml[
        [
            "team",
            "opponent",
            "moneyline_odds",
            "win_probability",
        ]
    ],
    use_container_width=True,
)

st.markdown("### Totals")
if not game_tot.empty:
    st.dataframe(
        game_tot[
            [
                "home_team",
                "away_team",
                "total_line",
                "predicted_total_points",
            ]
        ],
        use_container_width=True,
    )
else:
    st.write("No totals data.")

st.markdown("### Spread")
if not game_sp.empty:
    st.dataframe(
        game_sp[
            [
                "home_team",
                "away_team",
                "spread_line",
                "predicted_margin",
            ]
        ],
        use_container_width=True,
    )
else:
    st.write("No spread data.")

st.markdown("---")
st.markdown("### Recommended Moneyline Bets (with Actions)")

# Example: derive simple recs for ML
recs_full = game_ml.copy()
recs_full["win_probability"] = recs_full["win_probability"].apply(
    lambda x: x / 100.0 if x > 1 else x
)
recs_full["ml_edge"] = recs_full["win_probability"] - (
    recs_full["moneyline_odds"].apply(
        lambda o: 100 / (o + 100) if o > 0 else abs(o) / (abs(o) + 100)
    )
)

recs_full["confidence_index"] = (recs_full["ml_edge"] * 100).clip(lower=0)

if "parlay_legs" not in st.session_state:
    st.session_state["parlay_legs"] = []

for idx, row in recs_full.iterrows():
    team = row["team"]
    opponent = row["opponent"]
    price = row["moneyline_odds"]

    win_prob = float(row["win_probability"])  # 0–1
    win_prob_pct = win_prob * 100
    conf = row["confidence_index"]

    exp_label = f"{team} vs {opponent} — {team} ML ({conf:.1f} conf)"

    with st.expander(exp_label):
        st.write(f"Win probability: {win_prob_pct:.1f}%")
        st.write(f"Edge: {row['ml_edge']:.3f}")
        st.write(f"Odds: {price}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                f"➕ Add {team} ML to Parlay",
                key=f"gb_add_parlay_{selected_game}_{team}_{idx}",
            ):
                leg = ParlayLeg(
                    description=f"{team} ML ({team} vs {opponent})",
                    odds=float(price),
                    win_prob=win_prob,
                    team=team,
                    opponent=opponent,
                    market="moneyline",
                    source="game_breakdown",
                )
                st.session_state["parlay_legs"].append(leg)
                st.success("Added to parlay.")

        with col2:
            if st.button(
                f"📝 Log {team} ML bet",
                key=f"gb_log_bet_{selected_game}_{team}_{idx}",
            ):
                record = BetRecord(
                    bet_id=new_bet_id(),
                    date=str(date.today()),
                    game_date=str(pred_date),
                    market="moneyline",
                    team=team,
                    opponent=opponent,
                    bet_description=f"{team} ML (game breakdown)",
                    odds=float(price),
                    stake=100.0,
                    result="pending",
                    payout=0.0,
                    edge=float(row["ml_edge"]),
                    confidence="High" if conf >= 7 else "Medium" if conf >= 4 else "Low",
                    source="game_breakdown",
                )
                append_bet(record)
                st.success(f"Bet logged. Bet ID: {record.bet_id}")
