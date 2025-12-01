import streamlit as st
import pandas as pd
from app.predict_pipeline import predict_game  # your model inference function

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
            df.to_csv("results/custom_predictions.csv", mode="a", header=False, index=False)

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
                    results.append({
                        "date": row["date"],
                        "home_team": row["home_team"],
                        "away_team": row["away_team"],
                        "home_win_prob": prob_home,
                        "away_win_prob": 1 - prob_home
                    })
                except Exception as e:
                    results.append({
                        "date": row["date"],
                        "home_team": row["home_team"],
                        "away_team": row["away_team"],
                        "error": str(e)
                    })
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)

            st.download_button(
                label="Download Batch Predictions",
                data=results_df.to_csv(index=False),
                file_name="batch_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing CSV: {e}")