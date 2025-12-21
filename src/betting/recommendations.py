# ============================================================
# File: src/betting/recommendations.py
# ============================================================

import pandas as pd


def recommend_bets(bets_df, min_edge=0.03, min_ev=0.0, max_kelly=0.05, bankroll=100):
    if bets_df.empty:
        return bets_df

    df = bets_df.copy()

    df = df[df["edge"] >= min_edge]
    df = df[df["ev_per_unit"] >= min_ev]
    df = df[df["kelly_fraction"] <= max_kelly]

    if df.empty:
        return df

    df["recommended_stake"] = bankroll * df["kelly_fraction"]

    df["confidence"] = (
        df["edge"] * 50 + df["ev_per_unit"] * 30 + df["kelly_fraction"] * 20
    ) * 100

    df = df.sort_values("confidence", ascending=False)

    return df


def explain_confidence(row):
    reasons = []

    if row["edge"] > 0.10:
        reasons.append(
            "Model probability is significantly higher than market implied probability."
        )
    elif row["edge"] > 0.05:
        reasons.append(
            "Model probability is moderately higher than market implied probability."
        )
    else:
        reasons.append("Model shows a small advantage over the market.")

    if row["ev_per_unit"] > 0.05:
        reasons.append("Expected value per unit is strong.")
    elif row["ev_per_unit"] > 0.02:
        reasons.append("Expected value per unit is positive.")
    else:
        reasons.append("Expected value is marginal.")

    if row["kelly_fraction"] > 0.05:
        reasons.append("Kelly suggests an aggressive stake.")
    elif row["kelly_fraction"] > 0.02:
        reasons.append("Kelly suggests a moderate stake.")
    else:
        reasons.append("Kelly suggests a small stake.")

    return " ".join(reasons)
