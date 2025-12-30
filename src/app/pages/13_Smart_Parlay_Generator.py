from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Page: Smart Parlay Generator
# File: src/app/pages/13_Smart_Parlay_Generator.py
#
# Description:
#     Suggests parlays built from high-edge ML recommendations:
#       - uses automated recommendations engine for legs
#       - builds candidate parlays
#       - evaluates EV & Kelly
#       - allows pushing parlays into session for Parlay Builder
# ============================================================

from datetime import date
import itertools
import random

import numpy as np
import pandas as pd
import streamlit as st


from src.config.paths import ODDS_DIR
from src.model.predict import run_prediction_for_date
from src.model.predict_totals import run_totals_prediction_for_date
from src.model.predict_spread import run_spread_prediction_for_date
from src.markets.recommend import generate_recommendations, RecommendationConfig
from src.app.engines.parlay import ParlayLeg, parlay_expected_value
from src.app.engines.bet_tracker import kelly_fraction


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


if "parlay_legs" not in st.session_state:
    st.session_state["parlay_legs"] = []

pred_date = st.date_input("Prediction date", value=date.today())
min_edge = st.slider("Minimum ML edge (decimal)", 0.0, 0.2, 0.03, 0.01)
min_conf = st.slider("Minimum confidence index", 0, 100, 60, 5)
num_legs = st.slider("Number of legs per parlay", 2, 6, 3)
max_parlays = st.slider("Max suggested parlays", 1, 50, 10)

bankroll = st.number_input("Bankroll for Kelly sizing", value=1000.0, min_value=0.0)
stake = st.number_input("Stake to evaluate EV", value=100.0, min_value=0.0)

ml = run_prediction_for_date(pred_date)
totals = run_totals_prediction_for_date(pred_date)
spread = run_spread_prediction_for_date(pred_date)
odds = load_odds_for_date(pred_date)

if ml.empty:
    st.info("No games for this date.")
    st.stop()

cfg = RecommendationConfig(
    min_probability=0.0,
    min_edge=-1.0,
    min_stability=0.0,
)

recs = generate_recommendations(ml, totals, spread, odds, cfg=cfg)
recs = recs[
    (recs["ml_edge"] >= min_edge)
    & (recs["confidence_index"] >= min_conf)
    & recs["price"].notna()
].copy()

if recs.empty:
    st.warning("No recommendation legs meet the criteria.")
    st.stop()

st.markdown("### Candidate legs")
st.dataframe(
    recs[
        [
            "game_id",
            "team",
            "opponent",
            "win_probability",
            "price",
            "ml_edge",
            "stability_score",
            "confidence_index",
        ]
    ],
    use_container_width=True,
)

# Build candidate parlays
st.markdown("---")
st.markdown("### Suggested parlays")

legs = []
for _, row in recs.iterrows():
    legs.append(
        ParlayLeg(
            description=f"{row['team']} ML ({row['team']} vs {row['opponent']})",
            odds=float(row["price"]),
            win_prob=float(row["win_probability"]),
        )
    )

if len(legs) < num_legs:
    st.warning(f"Not enough legs ({len(legs)}) to build parlays of size {num_legs}.")
    st.stop()

# Strategy: mix of top-combos and random combos
all_indices = list(range(len(legs)))
greedy_indices = all_indices[: min(len(all_indices), 8)]  # top 8 legs
greedy_combos = list(itertools.combinations(greedy_indices, num_legs))
random_combos = []

while (
    len(random_combos) + len(greedy_combos) < max_parlays * 2
    and len(random_combos) < max_parlays * 3
):
    combo = tuple(sorted(random.sample(all_indices, num_legs)))
    if combo not in greedy_combos and combo not in random_combos:
        random_combos.append(combo)

all_combos = greedy_combos + random_combos
all_combos = all_combos[: max_parlays * 3]

parlays = []

for combo in all_combos:
    combo_legs = [legs[i] for i in combo]
    result = parlay_expected_value(combo_legs, stake)
    win_prob = result["win_prob"]
    dec_odds = result["decimal_odds"]
    ev = result["ev"]

    # Approximate Kelly for parlay
    b = dec_odds - 1.0
    q = 1 - win_prob
    kelly_f = (b * win_prob - q) / b if b > 0 else 0.0
    kelly_f = max(float(kelly_f), 0.0)

    parlays.append(
        {
            "legs": combo_legs,
            "win_prob": win_prob,
            "decimal_odds": dec_odds,
            "ev": ev,
            "kelly_fraction": kelly_f,
            "kelly_stake": kelly_f * bankroll,
        }
    )

parlays_df = pd.DataFrame(
    [
        {
            "description": " | ".join([leg.description for leg in p["legs"]]),
            "win_prob_pct": p["win_prob"] * 100,
            "decimal_odds": p["decimal_odds"],
            "ev": p["ev"],
            "kelly_fraction": p["kelly_fraction"],
            "kelly_stake": p["kelly_stake"],
        }
        for p in parlays
    ]
)

parlays_df = parlays_df.sort_values("ev", ascending=False).head(max_parlays)

st.dataframe(
    parlays_df[
        [
            "description",
            "win_prob_pct",
            "decimal_odds",
            "ev",
            "kelly_fraction",
            "kelly_stake",
        ]
    ],
    use_container_width=True,
)

st.markdown("### Actions")

for i, p in enumerate(parlays_df.itertuples()):
    with st.expander(f"Parlay {i+1}: EV={p.ev:.2f}, Kelly={p.kelly_fraction:.3f}"):
        st.write(p.description)
        st.write(f"Win probability: {p.win_prob_pct:.1f}%")
        st.write(f"Decimal odds: {p.decimal_odds:.3f}")
        st.write(f"Kelly stake: {p.kelly_stake:.2f} from bankroll {bankroll:.2f}")

        if st.button("➕ Use this parlay in Parlay Builder", key=f"spg_use_{i}"):
            # Replace session parlay_legs with these legs
            idx = parlays_df.index[i]
            combo_legs = parlays[i]["legs"]
            st.session_state["parlay_legs"] = combo_legs
            st.success(
                "Parlay legs loaded into session. Open Parlay Builder to inspect and log."
            )
