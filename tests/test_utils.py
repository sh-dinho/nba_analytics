import pandas as pd
from src.utils import add_unique_id

def test_add_unique_id():
    df = pd.DataFrame({"GAME_ID":[1], "TEAM_ID":[100], "prediction_date":["2025-12-09"]})
    df = add_unique_id(df)
    assert "unique_id" in df.columns
    assert df["unique_id"].nunique() == len(df)

def test_deduplication():
    df = pd.DataFrame({
        "GAME_ID":[1,1],
        "TEAM_ID":[100,100],
        "prediction_date":["2025-12-09","2025-12-09"]
    })
    df = add_unique_id(df).drop_duplicates(subset=["unique_id"])
    assert len(df) == 1
