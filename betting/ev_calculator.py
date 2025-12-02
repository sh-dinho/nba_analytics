# betting/ev_calculator.py
import pandas as pd

def calculate_ev(df):
    """
    Add expected value column
    EV = (prob * (odds - 1) - (1 - prob))
    """
    df["ev"] = df["pred_home_win_prob"] * (df["decimal_odds"] - 1) - (1 - df["pred_home_win_prob"])
    return df
