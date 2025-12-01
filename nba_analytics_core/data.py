# Path: nba_analytics_core/data.py
from nba_api.stats.endpoints import leaguegamefinder, scoreboardv2
import pandas as pd

def fetch_historical_games(season="2023-24"):
    """
    Fetch historical games for a season and derive outcomes.
    Returns: list of dicts {game_id, home_team, away_team, home_win}
    """
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
    df = gamefinder.get_data_frames()[0]

    results = []
    # leaguegamefinder is team-centric; this heuristic extracts home/away
    for _, row in df.iterrows():
        matchup = row.get("MATCHUP", "")
        team = row.get("TEAM_NAME", "")
        if "vs." in matchup:
            parts = matchup.split(" vs. ")
            home_team = parts[0]
            away_team = parts[1]
            home_win = row["WL"] == "W" if team == home_team else row["WL"] == "L"
            results.append({
                "game_id": row["GAME_ID"],
                "home_team": home_team,
                "away_team": away_team,
                "home_win": home_win
            })
    return results

def fetch_today_games():
    """
    Fetch today's scheduled games using scoreboardv2.
    Returns: list of dicts {home_team, away_team}
    """
    sb = scoreboardv2.ScoreboardV2()
    games = sb.game_header.get_data_frame()
    lineups = sb.line_score.get_data_frame()
    teams_map = {}
    for _, row in lineups.iterrows():
        teams_map[row["TEAM_ID"]] = row["TEAM_ABBREVIATION"]

    today_games = []
    for _, row in games.iterrows():
        home = row["HOME_TEAM_ID"]
        away = row["VISITOR_TEAM_ID"]
        home_team = teams_map.get(home)
        away_team = teams_map.get(away)
        if home_team and away_team:
            today_games.append({"home_team": home_team, "away_team": away_team})
    return today_games

def build_team_stats(games):
    """
    Build simple team-level stats from provided games list.
    """
    stats = {}
    for g in games:
        home = g["home_team"]
        away = g["away_team"]
        stats.setdefault(home, {"wins": 0, "losses": 0})
        stats.setdefault(away, {"wins": 0, "losses": 0})
        if "home_win" in g:
            if g["home_win"]:
                stats[home]["wins"] += 1
                stats[away]["losses"] += 1
            else:
                stats[home]["losses"] += 1
                stats[away]["wins"] += 1
    return stats

def build_matchup_features(home_team, away_team, team_stats):
    """
    Feature: win percentage difference (home minus away).
    """
    home = team_stats.get(home_team, {"wins": 0, "losses": 0})
    away = team_stats.get(away_team, {"wins": 0, "losses": 0})
    hg = home["wins"] + home["losses"]
    ag = away["wins"] + away["losses"]
    home_wpct = home["wins"] / hg if hg > 0 else 0.5
    away_wpct = away["wins"] / ag if ag > 0 else 0.5
    return [home_wpct - away_wpct]