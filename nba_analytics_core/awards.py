# Path: nba_analytics_core/awards.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def build_mvp_features(df):
    cols = [c for c in ["PTS","AST","REB","TS_PCT","MIN"] if c in df.columns]
    X = df[cols].fillna(0).values
    return X, df

def train_mvp_classifier(df_with_labels):
    """
    df_with_labels must include 'is_mvp' boolean and stats columns.
    """
    X, ctx = build_mvp_features(df_with_labels)
    y = df_with_labels["is_mvp"].astype(int).values
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(X, y)
    return model

def predict_mvp(df, model):
    X, ctx = build_mvp_features(df)
    probs = model.predict_proba(X)[:, 1]
    out = ctx[["PLAYER_NAME","TEAM_ABBREVIATION"]].copy()
    out["mvp_prob"] = probs
    return out.sort_values("mvp_prob", ascending=False)