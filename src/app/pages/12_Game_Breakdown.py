from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Page: Game Breakdown
# File: src/app/pages/12_Game_Breakdown.py
#
# Description:
#     Single-game deep dive:
#       - ML / Spread / Totals predictions
#       - Edges vs market
#       - Stability and confidence
#       - Direct actions: Log Bet, Add to Parlay
# ============================================================

from datetime import date

import pandas as pd
import streamlit as st

from src.markets.recommend import (
    generate_recommendations,
    RecommendationConfig,
)
from src.model.predict import run_prediction_for_date
from src.model.predict_totals import run_totals_prediction_for_date
from src.model.predict_spread import run_spread_prediction_for_date
from src.config.paths import ODDS_DIR
from src.app.engines.parlay import ParlayLeg
from src.app.engines.bet_tracker import BetRecord, append_bet, new_bet_id

from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar

render_header()
set_active_page("PAGE NAME HERE")
render_navbar()


def load_odds_for_date(pred_date: date) -> pd.DataFrame:
    base = f"odds_{pred_date.isoformat()}"
    parquet_path = ODDS_DIR / f"{base}.parquet"
    csv_path = ODDS_DIR / f"{base}.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    return pd.DataFrame(columns=["game_id", "price"])


pred_date = st.date_input("Prediction date", value=date.today())

ml = run_prediction_for_date(pred_date)
totals = run_totals_prediction_for_date(pred_date)
spread = run_spread_prediction_for_date(pred_date)
odds = load_odds_for_date(pred_date)

if ml.empty:
    st.info("No games for this date.")
    st.stop()

# Build a game list from ML predictions
ml["matchup"] = ml["team"] + " vs " + ml["opponent"]
games = ml[["game_id", "matchup"]].drop_duplicates()

game_id = st.selectbox(
    "Select game",
    games["game_id"],
    format_func=lambda gid: games.loc[games["game_id"] == gid, "matchup"].iloc[0],
)

# Filter per-game
ml_game = ml[ml["game_id"] == game_id].copy()
tot_game = (
    totals[totals["game_id"] == game_id].copy()
    if "game_id" in totals.columns
    else pd.DataFrame()
)
sp_game = (
    spread[spread["game_id"] == game_id].copy()
    if "game_id" in spread.columns
    else pd.DataFrame()
)
odds_game = odds[odds["game_id"] == game_id].copy()

# Use recommendations engine for ML edge + stability
cfg = RecommendationConfig(min_probability=0.0, min_edge=-1.0, min_stability=0.0)
recs_full = generate_recommendations(ml_game, tot_game, sp_game, odds_game, cfg=cfg)

st.markdown("### Matchup")

matchup_label = games.loc[games["game_id"] == game_id, "matchup"].iloc[0]
st.subheader(matchup_label)

col_left, col_right = st.columns(2)

# ML section
with col_left:
    st.markdown("#### Moneyline")

    if not ml_game.empty:
        st.dataframe(
            ml_game[
                [
                    "team",
                    "opponent",
                    "win_probability",
                    "moneyline_odds",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.info("No moneyline predictions for this game.")

# Totals / Spread section
with col_right:
    st.markdown("#### Totals / Spread")

    if not tot_game.empty and {"predicted_total_points", "total_line"} <= set(
        tot_game.columns
    ):
        tot_game = tot_game.copy()
        tot_game["diff"] = tot_game["predicted_total_points"] - tot_game["total_line"]
        st.markdown("**Totals**")
        st.dataframe(
            tot_game[
                [
                    "predicted_total_points",
                    "total_line",
                    "diff",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.info("No totals data for this game.")

    if not sp_game.empty and {"predicted_margin", "spread_line"} <= set(
        sp_game.columns
    ):
        sp_game = sp_game.copy()
        sp_game["edge"] = sp_game["predicted_margin"] - sp_game["spread_line"]
        st.markdown("**Spread**")
        st.dataframe(
            sp_game[
                [
                    "home_team",
                    "away_team",
                    "predicted_margin",
                    "spread_line",
                    "edge",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.info("No spread data for this game.")

st.markdown("---")
st.markdown("### ML Recommendations & Stability")

if recs_full.empty:
    st.info("No ML recommendations for this game.")
else:
    st.dataframe(
        recs_full[
            [
                "team",
                "opponent",
                "win_probability",
                "price",
                "ml_edge",
                "stability_score",
                "recommendation_score",
                "confidence_index",
                "recommendation",
            ]
        ],
        use_container_width=True,
    )

st.markdown("---")
st.markdown("### Actions")

if "parlay_legs" not in st.session_state:
    st.session_state["parlay_legs"] = []

stake_default = st.number_input(
    "Default stake for logging bets", value=100.0, step=10.0
)

for _, row in recs_full.iterrows():
    team = row["team"]
    opponent = row["opponent"]
    price = row["price"]
    win_prob = float(row["win_probability"])
    conf = row["confidence_index"]

    with st.expander(f"{team} vs {opponent} — {team} ML ({conf} conf)"):
        st.write(row["recommendation"])
        st.write(f"Win probability: {win_prob:.2f}")
        st.write(f"Edge: {row['ml_edge']:.3f}")
        st.write(f"Stability: {row['stability_score']:.2f}")
        st.write(f"Odds: {price if pd.notna(price) else 'n/a'}")

        col1, col2 = st.columns(2)

        with col1:
            can_add = pd.notna(price)
            if st.button(
                f"➕ Add {team} ML to Parlay",
                key=f"gb_add_parlay_{team}",
                disabled=not can_add,
            ):
                leg = ParlayLeg(
                    description=f"{team} ML ({team} vs {opponent})",
                    odds=float(price),
                    win_prob=win_prob,
                )
                st.session_state["parlay_legs"].append(leg)
                st.success("Added to parlay.")

        with col2:
            can_log = pd.notna(price)
            if st.button(
                f"📝 Log {team} ML bet", key=f"gb_log_bet_{team}", disabled=not can_log
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
                    stake=float(stake_default),
                    result="pending",
                    payout=0.0,
                    edge=float(row["ml_edge"]),
                    confidence=(
                        "High" if conf >= 80 else "Medium" if conf >= 60 else "Low"
                    ),
                )
                append_bet(record)
                st.success(f"Bet logged. Bet ID: {record.bet_id}")

st.markdown("---")
st.info(
    "You can see these bets and parlay legs in the Bet Tracker and Parlay Builder pages."
)
