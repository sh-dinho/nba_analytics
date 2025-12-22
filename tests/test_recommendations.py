from src.markets.recommend import generate_recommendations
import pandas as pd


def test_recommendations_generate():
    ml = pd.DataFrame(
        {"game_id": ["G1"], "team": ["A"], "opponent": ["B"], "win_probability": [0.65]}
    )
    totals = pd.DataFrame()
    spread = pd.DataFrame()
    odds = pd.DataFrame()

    recs = generate_recommendations(ml, totals, spread, odds)
    assert not recs.empty
    assert recs.iloc[0]["recommendation"] == "A ML"
