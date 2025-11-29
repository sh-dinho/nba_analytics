def generate_features(df):
    df = df.fillna(0)
    X = df[["PTS", "REB", "AST"]]
    return X
