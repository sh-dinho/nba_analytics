# ============================================================
# File: src/monitoring/team_risk.py
# ============================================================

import pandas as pd


def compute_team_risk(bet_log: pd.DataFrame):
    if bet_log.empty:
        return pd.DataFrame()

    df = bet_log.copy()

    df["error"] = (df["model_prob"] - df["actual_result"]).abs()

    df["upset"] = ((df["model_prob"] > 0.6) & (df["actual_result"] == 0)) | (
        (df["model_prob"] < 0.4) & (df["actual_result"] == 1)
    )

    risk = df.groupby("team").agg(
        error_mean=("error", "mean"),
        upset_rate=("upset", "mean"),
        volatility=("pnl", "std"),
    )

    risk = risk.fillna(0)

    if len(risk) > 1:
        risk_norm = (risk - risk.min()) / (risk.max() - risk.min()).replace(0, 1)
    else:
        risk_norm = risk.copy()

    risk_norm["risk_score"] = (
        risk_norm["error_mean"] + risk_norm["upset_rate"] + risk_norm["volatility"]
    ) / 3

    return risk_norm.sort_values("risk_score", ascending=False)
