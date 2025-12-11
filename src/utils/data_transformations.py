import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.dropna()  # Remove rows with NaN values
    return df

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(columns={
        "PTS": "POINTS",
        "WL": "TARGET",
    }, inplace=True)
    return df
