import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, balanced_accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from data import fetch_historical_games, engineer_features
from models import train_models
from simulation import simulate_bankroll

# ----------------------------
# Helper plots
# ----------------------------

def plot_confusion_matrix_with_metrics(
    y_test, y_pred,
    labels=("Home Loss", "Home Win"),
    normalize=False,
    cmap="Blues"
):
    cm = confusion_matrix(y_test, y_pred)
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
    ax.set_title(f"Confusion Matrix\nAcc: {acc*100:.2f}% | Balanced Acc: {bacc*100:.2f}%")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Metrics text
    lines = [f"Accuracy: {acc*100:.2f}% | Balanced Accuracy: {bacc*100:.2f}%"]
    for i, label in enumerate(labels):
        lines.append(f"{label}: Precision={precision[i]:.2f}, Recall={recall[i]:.2f}, F1={f1[i]:.2f}")
    st.text("\n".join(lines))

def plot_regression_evaluation(y_test, y_pred_points):
    mse = mean_squared_error(y_test, y_pred_points)
    rmse = mean_squared_error(y_test, y_pred_points, squared=False)
    mae = mean_absolute_error(y_test, y_pred_points)
    r2 = r2_score(y_test, y_pred_points)

    # Scatter plot
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.scatter(y_test, y_pred_points, alpha=0.6)
    lo = min(y_test.min(), y_pred_points.min())
    hi = max(y_test.max(), y_pred_points.max())
    ax1.plot([lo, hi], [lo, hi], color='red', lw=2)
    ax1.set_xlabel("True Total Points")
    ax1.set_ylabel("Predicted Total Points")
    ax1.set_title("True vs Predicted")
    st.pyplot(fig1)

    # Residuals plot
    residuals = y_test - y_pred_points
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.histplot(residuals, bins=30, kde=True, color="blue", ax=ax2)
    ax2.axvline(0, color="red", linestyle="--")
    ax2.set_xlabel("Residuals (True - Predicted)")
    ax2.set_title("Residuals Distribution")
    st.pyplot(fig2)

    st.text(f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}")

def plot_simulation_tab(
    probs,
    starting_bankroll=150,
    games=82,
    method="Fixed",
    confidence_factor=0.05,
    odds=1.8,
    min_stake=1.0
):
    # Use provided probs (truncate or pad to 'games')
    probs = list(probs)
    if len(probs) >= games:
        probs_use = probs[:games]
    else:
        # pad with mean prob if not enough
        mean_p = np.mean(probs) if len(probs) > 0 else 0.5
        probs_use = probs + [mean_p] * (games - len(probs))

    history, final_bankroll, roi, win_rate, max_drawdown = simulate_bankroll(
        probs_use, odds=odds, starting_bankroll=starting_bankroll,
        method=method, confidence_factor=confidence_factor, min_stake=min_stake
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history, label="Bankroll")
    ax.axhline(y=starting_bankroll, color="red", linestyle="--", label="Starting bankroll")
    ax.set_title("Simulated Bankroll Over Season")
    ax.set_xlabel("Games")
    ax.set_ylabel("Bankroll ($)")
    ax.legend()
    st.pyplot(fig)

    st.text(
        f"Final Bankroll: ${final_bankroll:.2f}\n"
        f"ROI: {roi:.2%}\n"
        f"Win Rate: {win_rate:.2%}\n"
        f"Max Drawdown: {max_drawdown:.2%}"
    )

# ----------------------------
# Streamlit app
# ----------------------------

def main():
    st.title("📊 NBA Model Evaluation & Simulation Dashboard")

    # Sidebar controls
    st.sidebar.header("Data & Training")
    season = st.sidebar.text_input("Season", value="2023")
    do_train = st.sidebar.button("Fetch, Engineer, and Train")

    st.sidebar.header("Confusion Matrix Options")
    normalize_cm = st.sidebar.checkbox("Show percentages (normalize rows)", value=True)
    cm_cmap = st.sidebar.selectbox("Colormap", options=["Blues", "Greens", "Purples", "Oranges"], index=0)

    st.sidebar.header("Simulation Controls")
    starting_bankroll = st.sidebar.slider("Starting bankroll ($)", min_value=50, max_value=2000, value=150, step=50)
    games = st.sidebar.slider("Games to simulate", min_value=20, max_value=100, value=82, step=2)
    strategy = st.sidebar.selectbox("Stake strategy", options=["Fixed", "Kelly"], index=0)
    confidence_factor = st.sidebar.slider("Fixed % confidence factor", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    odds = st.sidebar.slider("Decimal odds", min_value=1.5, max_value=2.5, value=1.8, step=0.05)
    min_stake = st.sidebar.slider("Minimum stake ($)", min_value=1, max_value=50, value=1, step=1)

    # State
    if "trained" not in st.session_state:
        st.session_state.trained = False
    if "clf" not in st.session_state:
        st.session_state.clf = None
    if "reg" not in st.session_state:
        st.session_state.reg = None
    if "eval_class" not in st.session_state:
        st.session_state.eval_class = None
    if "eval_reg" not in st.session_state:
        st.session_state.eval_reg = None

    # Training workflow
    if do_train:
        with st.spinner("Fetching games and training models..."):
            df_games = fetch_historical_games(season)
            df_feat = engineer_features(df_games)
            clf, reg, eval_class, eval_reg = train_models(df_feat)

            st.session_state.clf = clf
            st.session_state.reg = reg
            st.session_state.eval_class = eval_class  # (X_test_c, y_test_c, y_pred_c, y_proba_c)
            st.session_state.eval_reg = eval_reg      # (y_test_r, y_pred_r)
            st.session_state.trained = True

        st.success("Training complete.")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Classification", "Regression", "Simulation"])

    with tab1:
        st.subheader("Classification performance")
        if not st.session_state.trained:
            st.info("Train models to view classification results.")
        else:
            _, y_test_c, y_pred_c, _ = st.session_state.eval_class
            plot_confusion_matrix_with_metrics(
                y_test_c, y_pred_c,
                labels=("Home Loss", "Home Win"),
                normalize=normalize_cm,
                cmap=cm_cmap
            )

    with tab2:
        st.subheader("Regression performance")
        if not st.session_state.trained:
            st.info("Train models to view regression results.")
        else:
            y_test_r, y_pred_r = st.session_state.eval_reg
            plot_regression_evaluation(y_test_r, y_pred_r)

    with tab3:
        st.subheader("Simulation")
        if not st.session_state.trained:
            st.info("Train models to simulate bankroll with model-derived probabilities.")
        else:
            # Approximate per-game probabilities from classifier: use predicted probability on test set
            _, y_test_c, _, y_proba_c = st.session_state.eval_class
            # Clip probabilities to avoid extremes that break Kelly math
            probs = np.clip(y_proba_c, 0.05, 0.95)
            plot_simulation_tab(
                probs=probs,
                starting_bankroll=starting_bankroll,
                games=games,
                method=strategy,
                confidence_factor=confidence_factor,
                odds=odds,
                min_stake=min_stake
            )

if __name__ == "__main__":
    main()