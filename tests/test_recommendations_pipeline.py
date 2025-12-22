import pandas as pd
from src.markets.recommend import generate_recommendations


def test_recommendations_pipeline():
    ml = pd.DataFrame(
        {"game_id": ["G1"], "team": ["A"], "opponent": ["B"], "win_probability": [0.62]}
    )

    totals = pd.DataFrame(
        {
            "game_id": ["G1"],
            "home_team": ["A"],
            "away_team": ["B"],
            "predicted_total": [220],
        }
    )

    spread = pd.DataFrame(
        {
            "game_id": ["G1"],
            "home_team": ["A"],
            "away_team": ["B"],
            "predicted_margin": [5],
        }
    )

    odds = pd.DataFrame(
        {"game_id": ["G1"], "market_total": [215], "market_spread": [2]}
    )

    recs = generate_recommendations(ml, totals, spread, odds)
    assert not recs.empty
    assert "recommendation" in recs.columns
    assert recs.iloc[0]["confidence"] > 0
