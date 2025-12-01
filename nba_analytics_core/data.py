# File: nba_analytics_core/data.py

from nba_api.stats.endpoints import leaguegamefinder, scoreboardv2
import pandas as pd

def fetch_historical_games(season="2023-24"):
    """
    Fetch historical games for a season and derive outcomes.
    Returns: list of dicts {game_id, home_team, away_team, home_win}
    """
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        df = gamefinder.get_data_frames()[0]
    except Exception as e:
        print(f"Error fetching historical games: {e}")
        return []

    results = []
    for _, row in df.iterrows():
        matchup = row.get("MATCHUP", "")
        team = row.get("TEAM_NAME", "")
        wl = row.get("WL", "")

        if "vs." in matchup:  # home game
            home_team, away_team = matchup.split(" vs. ")
            home_win = (wl == "W") if team == home_team else (wl == "L")
            results.append({
                "game_id": row["GAME_ID"],
                "home_team": home_team,
                "away_team": away_team,
                "home_win": home_win
            })

        elif "@" in matchup:  # away game
            away_team, home_team = matchup.split(" @ ")
            home_win = (wl == "L") if team == away_team else (wl == "W")
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
    try:
        sb = scoreboardv2.ScoreboardV2()
        games = sb.game_header.get_data_frame()
        lineups = sb.line_score.get_data_frame()
    except Exception as e:
        print(f"Error fetching today's games: {e}")
        return []

    teams_map = {row["TEAM_ID"]: row["TEAM_ABBREVIATION"] for _, row in lineups.iterrows()}

    today_games = []
    for _, row in games.iterrows():
        home_team = teams_map.get(row["HOME_TEAM_ID"])
        away_team = teams_map.get(row["VISITOR_TEAM_ID"])
        if home_team and away_team:
            today_games.append({"home_team": home_team, "away_team": away_team})
    return today_games


def build_team_stats(games):
    """
    Build team-level stats from provided games list.
    Tracks wins, losses, point differential, and recent form (last 5 games).
    """
    stats = {}
    for g in games:
        home = g["home_team"]
        away = g["away_team"]
        stats.setdefault(home, {"wins": 0, "losses": 0, "point_diff": 0, "recent": []})
        stats.setdefault(away, {"wins": 0, "losses": 0, "point_diff": 0, "recent": []})

        if "home_win" in g:
            if g["home_win"]:
                stats[home]["wins"] += 1
                stats[away]["losses"] += 1
                stats[home]["point_diff"] += 1
                stats[away]["point_diff"] -= 1
                stats[home]["recent"].append(1)
                stats[away]["recent"].append(0)
            else:
                stats[home]["losses"] += 1
                stats[away]["wins"] += 1
                stats[home]["point_diff"] -= 1
                stats[away]["point_diff"] += 1
                stats[home]["recent"].append(0)
                stats[away]["recent"].append(1)

    # Limit recent form to last 5 games
    for team in stats:
        stats[team]["recent"] = stats[team]["recent"][-5:]
    return stats


def build_matchup_features(home_team, away_team, team_stats):
    """
    Features: win percentage difference, point differential, recent form difference.
    """
    home = team_stats.get(home_team, {"wins": 0, "losses": 0, "point_diff": 0, "recent": []})
    away = team_stats.get(away_team, {"wins": 0, "losses": 0, "point_diff": 0, "recent": []})

    hg = home["wins"] + home["losses"]
    ag = away["wins"] + away["losses"]
    home_wpct = home["wins"] / hg if hg > 0 else 0.5
    away_wpct = away["wins"] / ag if ag > 0 else 0.5
    diff_wpct = home_wpct - away_wpct

    diff_point = home["point_diff"] - away["point_diff"]

    home_recent = sum(home["recent"]) / len(home["recent"]) if home["recent"] else 0.5
    away_recent = sum(away["recent"]) / len(away["recent"]) if away["recent"] else 0.5
    diff_recent = home_recent - away_recent

    return [diff_wpct, diff_point, diff_recent]