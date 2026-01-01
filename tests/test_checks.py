from ingestion.validator.checks import find_asymmetry


def test_find_asymmetry_detects_mismatch():
    # Setup: Team A says they played B, but Team B says they played C
    df = pd.DataFrame([
        {"game_id": "1", "is_home": 1, "team": "LAL", "opponent": "BOS"},
        {"game_id": "1", "is_home": 0, "team": "BOS", "opponent": "GSW"}  # ERROR HERE
    ])

    bad_games = find_asymmetry(df)
    assert "1" in bad_games