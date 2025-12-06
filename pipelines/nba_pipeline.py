# ============================================================
# File: pipelines/nba_pipeline.py
# Purpose: Full NBA pipeline (data → features → train → predict → Telegram)
# ============================================================

import os
import sys
import logging
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv
from nba_api.stats.endpoints import leaguegamefinder

# ----------------------------------------------------------------
# Make sure project root is importable when run directly
# ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ----------------------------------------------------------------
# Project internal imports
# ----------------------------------------------------------------
from features.feature_builder import build_features
from train.train_model import train_model
from core.paths import (
    HISTORICAL_GAMES_FILE,
    NEW_GAMES_FILE,
    NEW_GAMES_FEATURES_FILE,
    ensure_dirs,
)
from core.config import (
    DEFAULT_BANKROLL,
    MAX_KELLY_FRACTION,
    EV_THRESHOLD,
    MIN_KELLY_STAKE,
)
from core.log_config import init_global_logger

# Initialize logger
logger = init_global_logger()

# ============================================================
# Environment
# ============================================================
load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not ODDS_API_KEY:
    raise ValueError("❌ Missing ODDS_API_KEY in .env")
if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
    logger.warning("⚠️ Telegram is not configured — notifications disabled")

ODDS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

# ============================================================
# Telegram
# ============================================================
def send_telegram_message(msg: str):
    """Send a formatted Telegram message."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram not configured")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown"
    }

    try:
        r = requests.post(url, json=payload, timeout=5)
        if r.ok:
            logger.info("📨 Telegram message sent.")
        else:
            logger.warning(f"⚠️ Telegram failed: {r.text}")
    except Exception as e:
        logger.error(f"❌ Telegram error: {e}")

# ============================================================
# Matchup Parsing
# ============================================================
def parse_matchup(matchup: str):
    """Convert MATCHUP string into (home_team, away_team)."""
    if matchup is None:
        return None, None

    if " vs. " in matchup or " vs " in matchup:
        x = matchup.replace(" vs. ", " vs ").split(" vs ")
        return x[0], x[1]

    if "@" in matchup:
        away, home = matchup.split(" @ ")
        return home, away

    return None, None

# ============================================================
# Load Season Data
# ============================================================
def load_season_data(season_label: str):
    logger.info(f"📅 Loading NBA season data: {season_label}")

    try:
        gf = leaguegamefinder.LeagueGameFinder(season_nullable=season_label)
        games_df = gf.get_data_frames()[0]
        logger.info(f"   → Loaded {len(games_df)} games")
    except Exception as e:
        logger.error(f"❌ NBA API error: {e}")
        return pd.DataFrame(), pd.DataFrame()

    games_df["GAME_DATE"] = pd.to_datetime(games_df["GAME_DATE"])
    games_df["home_team"], games_df["away_team"] = zip(
        *games_df["MATCHUP"].map(parse_matchup)
    )

    try:
        response = requests.get(
            ODDS_URL,
            params={"apiKey": ODDS_API_KEY, "regions": "us", "markets": "h2h,spreads,totals"},
            timeout=10
        )
        response.raise_for_status()
        odds_raw = response.json()

        odds_flat = []
        for game in odds_raw:
            for book in game.get("bookmakers", []):
                for market in book.get("markets", []):
                    odds_flat.append({
                        "id": game.get("id"),
                        "home_team": game.get("home_team"),
                        "away_team": game.get("away_team"),
                        "market": market.get("key"),
                        "bookmaker": book.get("title"),
                                                "outcomes": market.get("outcomes"),
                    })

        odds_df = pd.DataFrame(odds_flat)
        logger.info(f"   → Retrieved {len(odds_df)} odds rows")

    except Exception as e:
        logger.error(f"❌ Odds API error: {e}")
        odds_df = pd.DataFrame()

    return games_df, odds_df

# ============================================================
# Save Upcoming Games
# ============================================================
def save_new_games(games_df: pd.DataFrame, odds_df: pd.DataFrame):
    today = pd.Timestamp.today().normalize()
    upcoming = games_df[games_df["GAME_DATE"] >= today]

    if upcoming.empty:
        logger.warning("⚠️ No upcoming games found")
        return

    if not odds_df.empty:
        upcoming = upcoming.merge(
            odds_df,
            on=["home_team", "away_team"],
            how="left"
        )

    NEW_GAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    upcoming.to_csv(NEW_GAMES_FILE, index=False)
    logger.info(f"Saved upcoming games → {NEW_GAMES_FILE}")

# ============================================================
# Telegram Message Builder
# ============================================================
def generate_telegram_message(preds: pd.DataFrame, top_n: int = 5):
    if preds.empty:
        return "⚠️ No predictions available today."

    top = preds.sort_values("ev", ascending=False).head(top_n)

    msg = ["🏀 *Top NBA Picks Today*"]

    for _, r in top.iterrows():
        msg.append(
            f"*{r['home_team']}* vs *{r['away_team']}*\n"
            f"• Predicted Winner: *{r['predicted_winner']}*\n"
            f"• EV: `{r['ev']:.2f}` | Stake: `${r['stake']:.2f}`\n"
        )

    return "\n".join(msg)

# ============================================================
# Main Pipeline
# ============================================================
def run_pipeline_with_notifications(season_label: str):
    ensure_dirs()

    # 1) Load Data
    games_df, odds_df = load_season_data(season_label)
    if games_df.empty:
        logger.warning("⚠️ No historical games loaded — stopping.")
        return

    games_df.to_csv(HISTORICAL_GAMES_FILE, index=False)
    logger.info(f"Historical games saved → {HISTORICAL_GAMES_FILE}")

    # 2) Build training features
    build_features(training=True, player=True)

    # 3) Train models (Option 1 fix)
    team_metrics = train_model(model_type="team")
    player_metrics = train_model(model_type="player")

    logger.info(f"Team model metrics: {team_metrics}")
    logger.info(f"Player model metrics: {player_metrics}")

    # 4) Prepare upcoming games
    save_new_games(games_df, odds_df)
    build_features(training=False, player=False)

    # 5) Prediction step
    if not NEW_GAMES_FEATURES_FILE.exists():
        logger.warning("⚠️ No new game features found.")
        return

    preds = pd.read_csv(NEW_GAMES_FEATURES_FILE)

    # TODO: Replace with real model inference
    preds["predicted_winner"] = preds["home_team"]
    preds["ev"] = 8.7
    preds["stake"] = 25.0

    # 6) Telegram output
    msg = generate_telegram_message(preds)
    send_telegram_message(msg)

# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    SEASON = "2025-26"
    logger.info(f"🏀 Running NBA Pipeline for {SEASON}")
    run_pipeline_with_notifications(SEASON)