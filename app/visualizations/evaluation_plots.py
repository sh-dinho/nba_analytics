import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score

def plot_confusion_matrix_with_metrics(y_test, y_pred, labels, normalize=False, cmap="Blues"):
    """
    Plot a confusion matrix with optional normalization and display accuracy.
    """
    cm = confusion_matrix(y_test, y_pred)
    if normalize:
        cm = cm.astype('float')
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm, annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

    accuracy = (y_test == y_pred).mean()
    st.write(f"Accuracy: {accuracy:.2%}")

def plot_regression_evaluation(y_true, y_pred):
    """
    Plot regression evaluation: scatter plot of predictions vs true values,
    and residuals distribution.
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Scatter plot
    ax[0].scatter(y_true, y_pred, alpha=0.3)
    ax[0].plot([y_true.min(), y_true.max()],
               [y_true.min(), y_true.max()],
               'k--', lw=2)
    ax[0].set_xlabel('True Values')
    ax[0].set_ylabel('Predicted Values')
    ax[0].set_title(f"Regression: MSE = {mse:.2f}, R² = {r2:.2f}")

    # Residuals distribution
    residuals = y_true - y_pred
    sns.histplot(residuals, bins=50, kde=True, ax=ax[1], color="skyblue")
    ax[1].axvline(0, color='red', linestyle='--')
    ax[1].set_xlabel("Residuals")
    ax[1].set_title("Residuals Distribution")

    st.pyplot(fig)
    st.write(f"MSE: {mse:.2f}, R²: {r2:.2f}")

def plot_simulation_results(history, starting_bankroll, final_bankroll, roi, win_rate, max_drawdown):
    """
    Plot bankroll progression over time and annotate key metrics.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    color = "green" if final_bankroll >= starting_bankroll else "red"
    ax.plot(history, color=color)
    ax.set_title("Bankroll Over Time")
    ax.set_xlabel("Game Number")
    ax.set_ylabel("Bankroll")
    ax.grid(True)

    # Annotate start and end
    ax.text(len(history)-1, history[-1],
            f"${final_bankroll:.2f}", ha='right', va='bottom',
            fontsize=10, color=color)
    ax.text(0, starting_bankroll, f"Start: ${starting_bankroll:.2f}", fontsize=9, color="black")

    # Annotate max drawdown
    ax.text(len(history)//2, max(history), f"Max Drawdown: {max_drawdown:.2%}", fontsize=9, color="red")

    st.pyplot(fig)
    st.write(f"ROI: {roi:.2%}, Win Rate: {win_rate:.2%}, Max Drawdown: {max_drawdown:.2%}")