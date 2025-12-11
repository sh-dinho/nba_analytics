# ============================================================
# File: tests/test_mapping.py
# Purpose: Validate team and player ID mapping utilities
# Project: nba_analysis
# ============================================================

import pandas as pd
import pytest

from src.utils.mapping import map_team_ids, map_player_ids, map_ids


def test_map_team_ids_basic():
    df = pd.DataFrame({"TEAM_ID": [0, 29, 99]})
    out = map_team_ids(df)
    assert out.loc[0, "TEAM_NAME"] == "Team_0"
    assert out.loc[1, "TEAM_NAME"] == "Team_29"
    assert out.loc[2, "TEAM_NAME"] == "UnknownTeam"


def test_map_player_ids_basic():
    df = pd.DataFrame({"PLAYER_ID": [0, 999, 1001]})
    out = map_player_ids(df)
    assert out.loc[0, "PLAYER_NAME"] == "Player_0"
    assert out.loc[1, "PLAYER_NAME"] == "Player_999"
    assert out.loc[2, "PLAYER_NAME"] == "UnknownPlayer"


def test_map_ids_combined():
    df = pd.DataFrame({"TEAM_ID": [0, 29, 99], "PLAYER_ID": [0, 999, 1001]})
    out = map_ids(df)
    # TEAM_NAME and PLAYER_NAME should both exist
    assert "TEAM_NAME" in out.columns
    assert "PLAYER_NAME" in out.columns
    # Check known mappings
    assert out.loc[0, "TEAM_NAME"] == "Team_0"
    assert out.loc[0, "PLAYER_NAME"] == "Player_0"
    # Check unknowns
    assert out.loc[2, "TEAM_NAME"] == "UnknownTeam"
    assert out.loc[2, "PLAYER_NAME"] == "UnknownPlayer"


def test_map_ids_custom_maps():
    custom_team_map = {100: "CustomTeam"}
    custom_player_map = {2000: "CustomPlayer"}
    df = pd.DataFrame({"TEAM_ID": [100], "PLAYER_ID": [2000]})
    out = map_ids(df, team_map=custom_team_map, player_map=custom_player_map)
    assert out.loc[0, "TEAM_NAME"] == "CustomTeam"
    assert out.loc[0, "PLAYER_NAME"] == "CustomPlayer"


def test_invalid_input_raises():
    with pytest.raises(TypeError):
        map_team_ids([1, 2, 3])
    with pytest.raises(TypeError):
        map_player_ids([1, 2, 3])
    with pytest.raises(TypeError):
        map_ids([1, 2, 3])
