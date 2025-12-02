# betting/picks.py
import pandas as pd

def generate_picks(preds_df, threshold=0.6):
    picks = preds_df[preds_df["pred_home_win_prob"] >= threshold].copy()
    picks["pick"] = "home"
    return picks
