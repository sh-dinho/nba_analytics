import pandas as pd


def prepare_features(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for ML model.
    Example: home/away win rates, team stats, rest days, etc.
    """
    if schedule_df.empty:
        return pd.DataFrame()

    # Example: create numeric features from basic stats
    df = schedule_df.copy()
    df["home_team_encoded"] = df["hTeam.triCode"].astype("category").cat.codes
    df["away_team_encoded"] = df["vTeam.triCode"].astype("category").cat.codes
    df["home_score"] = pd.to_numeric(df.get("hTeam.score", 0))
    df["away_score"] = pd.to_numeric(df.get("vTeam.score", 0))
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    feature_cols = [
        "home_team_encoded",
        "away_team_encoded",
        "home_score",
        "away_score",
    ]
    features = df[feature_cols + ["home_win"]].copy()
    return features
