# Path: nba_analytics_core/awards.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

REQUIRED_FEATURES = ["PTS", "AST", "REB", "TS_PCT", "MIN"]

def build_mvp_features(df: pd.DataFrame):
    """Extract MVP candidate features from dataframe."""
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for MVP features: {missing}")
    X = df[REQUIRED_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).values
    return X, df

def train_mvp_classifier(df_with_labels: pd.DataFrame):
    """
    Train MVP classifier.
    df_with_labels must include 'is_mvp' boolean and stats columns.
    """
    if "is_mvp" not in df_with_labels.columns:
        raise ValueError("Training data must include 'is_mvp' column.")

    X, _ = build_mvp_features(df_with_labels)
    y = df_with_labels["is_mvp"].astype(int).values

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(X, y)

    metadata = {
        "n_samples": len(df_with_labels),
        "n_features": len(REQUIRED_FEATURES),
        "features": REQUIRED_FEATURES,
        "class_distribution": dict(pd.Series(y).value_counts())
    }

    return model, metadata

def predict_mvp(df: pd.DataFrame, model, top_n: int = None):
    """
    Predict MVP probabilities for players.
    Returns dataframe sorted by probability.
    """
    X, ctx = build_mvp_features(df)
    probs = model.predict_proba(X)[:, 1]

    out = ctx[["PLAYER_NAME", "TEAM_ABBREVIATION"]].copy()
    out["mvp_prob"] = probs
    out = out.sort_values("mvp_prob", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1

    if top_n:
        out = out.head(top_n)

    return out