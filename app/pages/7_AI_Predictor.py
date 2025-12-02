# ============================================================
# File: 7_AI_Predictor.py
# Path: <project_root>/pages/7_AI_Predictor.py
#
# Description:
#   Streamlit page for predicting NBA game outcomes using the 
#   trained model. Supports single game input via form or 
#   batch CSV uploads. Includes metrics, distribution charts,
#   filtering, and CSV export functionality.
# ============================================================

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
from app.predict_pipeline import predict_game  # your model inference function
import os

st.title("🤖 AI Predictor")
st.caption("Run the trained model on custom inputs or batch CSV uploads.")

# --- Single Prediction Form ---
st.subheader("Single Game Prediction")
with st.form("predict_form"):
    home_team = st.text_input("Home Team", "")
    away_team = st.text_input("Away Team", "")
    date = st.date_input("Game Date")
    submitted = st.form_submit_button("Predict")

if submitted:
    if not home_team or not away_team:
        st.error("Please enter both home and away teams.")
    else:
        try:
            prediction = predict_game({
                "date": str(date),
                "home_team": home_team,
                "away_team": away_team
            })
            if not (0 <= prediction <= 1):
                st.error("Model returned invalid probability.")
            else:
                st.success("Prediction complete!")
                st.metric("Home Win Probability", f"{prediction:.2%}")
                st.metric("Away Win Probability", f"{1 - prediction:.2%}")

                df = pd.DataFrame([{
                    "date": str(date),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_win_prob": prediction,
                    "away_win_prob": 1 - prediction
                }])

                # Append safely with header if file doesn't exist
                out_path = "results/custom_predictions.csv"
                df.to_csv(out_path, mode="a",
                          header=not os.path.exists(out_path),
                          index=False)

                st.download_button(
                    label="Download Prediction",
                    data=df.to_csv(index=False),
                    file_name="custom_prediction.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --- Batch Prediction ---
st.subheader("Batch Predictions (CSV Upload)")
st.markdown("Upload a CSV with columns: `date`, `home_team`, `away_team`.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        if not {"date","home_team","away_team"}.issubset(input_df.columns):
            st.error("CSV must contain columns: date, home_team, away_team")
        else:
            results = []
            for _, row in input_df.iterrows():
                try:
                    prob_home = predict_game({
                        "date": str(row["date"]),
                        "home_team": row["home_team"],
                        "away_team": row["away_team"]
                    })
                    if not (0 <= prob_home <= 1):
                        raise ValueError("Invalid probability returned")
                    results.append({
                        "date": row["date"],
                        "home_team": row["home_team"],
                        "away_team": row["away_team"],
                        "home_win_prob": prob_home,
                        "away_win_prob": 1 - prob_home
                    })
                except Exception as e:
                    results.append({
                        "date": row.get("date"),
                        "home_team": row.get("home_team"),
                        "away_team": row.get("away_team"),
                        "error": str(e)
                    })
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)

            # Sidebar filter for threshold
            threshold = st.sidebar.slider("Filter by minimum home win probability", 0.0, 1.0, 0.5, 0.05)
            filtered_df = results_df[results_df.get("home_win_prob", 0) >= threshold]

            # Summary metrics
            if "home_win_prob" in results_df.columns:
                st.subheader("Batch Summary")
                st.metric("Games Predicted", len(results_df))
                st.metric("Average Home Win Probability", f"{results_df['home_win_prob'].mean():.2%}")
                st.metric("Games Above Threshold", len(filtered_df))

                # Distribution chart
                chart = alt.Chart(results_df.dropna()).mark_bar().encode(
                    x=alt.X("home_win_prob", bin=alt.Bin(maxbins=30), title="Home Win Probability"),
                    y="count()",
                    tooltip=["home_team","away_team","home_win_prob"]
                )
                st.altair_chart(chart, use_container_width=True)

            # Export options
            st.download_button(
                label="Download Batch Predictions",
                data=results_df.to_csv(index=False),
                file_name="batch_predictions.csv",
                mime="text/csv"
            )
            st.download_button(
                label="Download Filtered Predictions",
                data=filtered_df.to_csv(index=False),
                file_name=f"batch_predictions_filtered_{threshold}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing CSV: {e}")
