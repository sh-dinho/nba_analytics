import streamlit as st

# Core modules
from core.data import fetch_historical_games, engineer_features
from core.db_module import connect
from core.fetch_games import get_todays_games
from core.team_performance import team_stats
from core.utils import send_telegram_message

# Models
from models.train_models import train_models_cached

# App modules
from app.predictor import predict_todays_games
from app.simulate_ai_bankroll import simulate_ai_strategy
from app.visualizations.evaluation_plots import plot_evaluation
from app.visualizations.visualizations import plot_dashboard
from app.visualizations.team_visualizations import plot_team


def main():
    st.title("🏀 NBA Analytics Dashboard")

    st.sidebar.header("Navigation")
    choice = st.sidebar.radio("Go to", ["Today's Games", "AI Predictions", "Bankroll Simulation", "Visualizations"])

    if choice == "Today's Games":
        games = get_todays_games()
        st.write(games)

    elif choice == "AI Predictions":
        preds = predict_todays_games()
        st.write(preds)

    elif choice == "Bankroll Simulation":
        sim = simulate_ai_strategy(initial_bankroll=1000, strategy="kelly")
        st.line_chart(sim["bankroll"])

    elif choice == "Visualizations":
        st.subheader("Evaluation Plots")
        plot_evaluation()
        st.subheader("Dashboard")
        plot_dashboard()
        st.subheader("Team Visualizations")
        plot_team("Lakers")


if __name__ == "__main__":
    main()