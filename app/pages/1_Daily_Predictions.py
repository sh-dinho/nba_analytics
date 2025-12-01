import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from datetime import datetime
from nba_analytics_core.odds import fetch_odds
from scripts.simulate_bankroll import simulate_bankroll
from app.predict_pipeline import generate_today_predictions

def highlight_ev(val):
    if pd.isna(val):
        return ""
    return "background-color: lightgreen" if val > 0 else "background-color: salmon"

st.title("📅 Daily Predictions")
st.caption("Win probabilities for today's games using the trained model + bookmaker odds.")

try:
    df = generate_today_predictions()
    if df.empty:
        st.info("No games found today or model not trained yet.")
    else:
        # Fetch odds
        odds_data = []
        for _, row in df.iterrows():
            odds = fetch_odds(home_team=row["home_team"], away_team=row["away_team"])
            odds_data.append({
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_win_prob": row["home_win_prob"],
                "home_odds": odds.get("home_odds") if odds else None,
                "away_odds": odds.get("away_odds") if odds else None,
                "bookmaker": odds.get("bookmaker") if odds else None,
                "last_update": odds.get("last_update") if odds else None
            })
        df_odds = pd.DataFrame(odds_data)

        # EV calculations
        for col in ["home_odds", "away_odds"]:
            df_odds[col] = pd.to_numeric(df_odds[col], errors="coerce")
        df_odds["ev_home"] = np.where(
            df_odds["home_odds"].notna(),
            df_odds["home_win_prob"] * df_odds["home_odds"] - (1 - df_odds["home_win_prob"]),
            np.nan
        )
        df_odds["ev_away"] = np.where(
            df_odds["away_odds"].notna(),
            (1 - df_odds["home_win_prob"]) * df_odds["away_odds"] - df_odds["home_win_prob"],
            np.nan
        )

        # EV summary
        num_positive_ev = ((df_odds["ev_home"] > 0) | (df_odds["ev_away"] > 0)).sum()
        st.metric("Positive EV Bets Today", num_positive_ev)

        # EV distribution chart
        ev_data = df_odds.melt(id_vars=["home_team","away_team"], value_vars=["ev_home","ev_away"], var_name="side", value_name="ev")
        chart_ev = alt.Chart(ev_data).mark_bar().encode(
            x=alt.X("ev", bin=alt.Bin(maxbins=30), title="Expected Value"),
            y="count()",
            color="side",
            tooltip=["home_team","away_team","ev"]
        )
        st.altair_chart(chart_ev, use_container_width=True)

        # Styled table
        styled_df = df_odds.style.applymap(highlight_ev, subset=["ev_home", "ev_away"])
        st.dataframe(styled_df, use_container_width=True)

        # Bankroll simulation preview
        sim_df = df_odds.rename(columns={"home_win_prob":"prob", "home_odds":"decimal_odds"})
        sim_input = sim_df[["decimal_odds","prob","ev_home"]].dropna()
        if not sim_input.empty:
            records, metrics = simulate_bankroll(sim_input, strategy="kelly")
            st.write(f"Final Bankroll: ${metrics['final_bankroll']:.2f}")
            st.write(f"ROI: {metrics['roi']*100:.2f}%")
            st.write(f"Win Rate: {metrics['win_rate']*100:.2f}% ({metrics['wins']}W/{metrics['losses']}L)")

            # Trajectory chart
            trajectory_df = pd.DataFrame({"Bet #": range(len(records)), "Bankroll": records})
            chart_bankroll = alt.Chart(trajectory_df).mark_line(point=True).encode(
                x="Bet #",
                y="Bankroll",
                tooltip=["Bet #","Bankroll"]
            ).interactive()
            st.altair_chart(chart_bankroll, use_container_width=True)

        # Export
        export_name = f"daily_predictions_{datetime.now().date()}.csv"
        st.download_button(
            label="Download Predictions as CSV",
            data=df_odds.to_csv(index=False),
            file_name=export_name,
            mime="text/csv"
        )

except Exception as e:
    st.error(f"Error generating predictions: {e}")