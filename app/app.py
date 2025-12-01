# app/app.py (Updated to use KPI summary)
import streamlit as st
import pandas as pd

# Core modules
from nba_core.data import fetch_historical_games, engineer_features
from nba_core.db_module import connect
from nba_core.fetch_games import get_todays_games
# from nba_core.team_performance import team_stats
# from nba_core.utils import send_telegram_message

# Models
from models.train_models import train_models_cached

# App modules
from app.predictor import predict_todays_games
from scripts.simulate_bankroll import simulate_bankroll
# from app.visualizations.evaluation_plots import plot_evaluation
# from app.visualizations.visualizations import plot_dashboard
# from app.visualizations.team_visualizations import plot_team

# Placeholder function for simulate_ai_strategy using historical data
def simulate_ai_strategy(initial_bankroll=1000, strategy="kelly"):
    # NOTE: This implementation relies on a function that loads and predicts on historical data
    # Mock data is used for display purposes, but in reality, this would run 
    # the backtesting pipeline to get a real sim_df with attached KPIs.
    
    # Mock history DataFrame
    data = {
        'bankroll': [1000, 1010, 990, 1030, 1050],
        'date': pd.to_datetime(['2025-11-20', '2025-11-21', '2025-11-22', '2025-11-23', '2025-11-24'])
    }
    sim_df = pd.DataFrame(data)
    
    # Attach mock KPIs matching the new structure for demonstration
    sim_df.kpis = {
        "roi": 0.05,
        "win_rate": 0.65,
        "max_drawdown": 0.04,
        "final_bankroll": 1050.00,
        "total_bets": 100
    }
    return sim_df


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
        st.subheader("Bankroll Simulation Results (Historical Backtest)")
        sim = simulate_ai_strategy(initial_bankroll=1000, strategy="kelly")
        
        # --- IMPROVEMENT: Display KPIs ---
        if not sim.empty and hasattr(sim, 'kpis'):
            kpis = sim.kpis
            
            # Display KPIs using Streamlit metrics for a clean look
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Final Bankroll", f"${kpis['final_bankroll']:.2f}")
            col2.metric("ROI", f"{kpis['roi']:.2%}")
            col3.metric("Win Rate", f"{kpis['win_rate']:.2%}")
            col4.metric("Max Drawdown", f"{kpis['max_drawdown']:.2%}")
            
            st.write(f"Total Realized Bets: {kpis['total_bets']}")
            st.line_chart(sim["bankroll"])
        else:
            st.warning("No simulation data or KPIs available.")


    elif choice == "Visualizations":
        st.subheader("Evaluation Plots")
        st.write("Visualizations are placeholders; assuming `plot_evaluation` and others consume saved artifacts.")
        st.write("---")
        st.subheader("Dashboard")
        st.write("---")
        st.subheader("Team Visualizations")


if __name__ == "__main__":
    main()