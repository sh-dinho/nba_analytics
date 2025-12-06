# ============================================================
# Telegram reporting utilities for NBA Analytics
# ============================================================

import re
import requests
from nba_core.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SEND_NOTIFICATIONS
from nba_core.log_config import init_global_logger

logger = init_global_logger("telegram")

# ---------------- Markdown Escaping ----------------
def escape_markdown(text: str) -> str:
    """
    Escape special characters for Telegram MarkdownV2.
    """
    return re.sub(r'([_*[\]()~`>#+-=|{}.!])', r'\\\1', text)

# ---------------- Confidence Scoring ----------------
def confidence_level(prob: float) -> str:
    if prob >= 0.70:
        return "🔥 High confidence"
    elif prob >= 0.60:
        return "⚖️ Medium confidence"
    else:
        return "❄️ Low confidence"

# ---------------- Headline Generator ----------------
def generate_headline(game_name: str, team_name: str, best_model: str, prob: float) -> str:
    if best_model == "team":
        return f"🔥 {team_name} favored strongly in {game_name}"
    elif best_model == "ml":
        return f"⚖️ Spread edge spotted: {game_name}"
    elif best_model == "ou":
        return f"📊 Totals play: {game_name} trending {'Over' if prob > 0.5 else 'Under'}"
    elif best_model == "player":
        return f"⭐ Player prop value in {game_name}"
    else:
        return f"🎲 Value detected in {game_name}"

# ---------------- Formatter ----------------
def format_top_picks_for_telegram(df, top_n=5):
    """
    Format DataFrame of predictions into a Telegram-friendly MarkdownV2 message.
    """
    stake_cols = [c for c in df.columns if c.startswith("stake_")]
    df["max_stake"] = df[stake_cols].max(axis=1)
    df["best_model"] = df[stake_cols].idxmax(axis=1).str.replace("stake_", "")

    top_picks = df.sort_values("max_stake", ascending=False).head(top_n)

    lines = ["🎯 *Top Picks (by Kelly stake)*\n"]
    for _, row in top_picks.iterrows():
        game_name = row.get("GAME_NAME") or f"Game {row.get('GAME_ID', '')}"
        team_name = row.get("TEAM_NAME", row.get("TEAM_ID", ""))

        best_model = row["best_model"]
        best_stake = row["max_stake"]

        # Probabilities
        prob_team = row.get("prob_team_win", 0)
        prob_player = row.get("prob_player_win", 0)
        prob_ml = row.get("prob_ml_win", 0)
        prob_ou = row.get("prob_ou_win", 0)

        # Guidance
        if best_stake > 0 and row.get(f"prob_{best_model}_win", 0) > 0.55:
            guidance = "🟢 *Recommended*: Stake (Kelly fraction positive, edge vs odds)"
        else:
            guidance = "🔴 *Avoid*: No clear edge (probability too low or Kelly fraction ≈ 0)"

        # Bet type explanation
        if best_model == "team":
            bet_type = "Moneyline (Team win)"
        elif best_model == "player":
            bet_type = "Player prop (PTS/REB/AST performance)"
        elif best_model == "ml":
            bet_type = "Spread bet (covering margin)"
        elif best_model == "ou":
            bet_type = "Over/Under total points"
        else:
            bet_type = "General outcome"

        # Confidence score
        conf = confidence_level(row.get(f"prob_{best_model}_win", 0))

        # Headline
        headline = generate_headline(game_name, team_name, best_model, row.get(f"prob_{best_model}_win", 0))

        # Avoid note: weakest model
        weakest_model = min(
            ["team", "player", "ml", "ou"],
            key=lambda m: row.get(f"prob_{m}_win", 0)
        )
        avoid_note = f"🔴 Avoid: {weakest_model.capitalize()} model (low probability)"

        lines.append(
            f"{headline}\n"
            f"🏀 {game_name} | {team_name}\n"
            f"👉 Best model: *{best_model.capitalize()}*\n"
            f"💰 Stake: {best_stake:.2f} units\n"
            f"🎲 Bet type: {bet_type}\n"
            f"{guidance}\n"
            f"{conf}\n"
            f"{avoid_note}\n"
            f"• Prob(team)={prob_team:.2f}, Prob(player)={prob_player:.2f}, "
            f"Prob(ml)={prob_ml:.2f}, Prob(ou)={prob_ou:.2f}\n"
            f"• Decimal odds={row.get('decimal_odds', 'N/A')}\n"
        )

    return "\n".join(lines)

# ---------------- Sender with Fallback ----------------
def send_telegram_message(text: str):
    """
    Send a message to Telegram with MarkdownV2 escaping.
    If Markdown fails, retry with plain text.
    """
    if not SEND_NOTIFICATIONS:
        logger.info("🔕 Notifications disabled (SEND_NOTIFICATIONS=False)")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    safe_text = escape_markdown(text)
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": safe_text,
        "parse_mode": "MarkdownV2"
    }

    try:
        r = requests.post(url, json=payload, timeout=10)
        resp = r.json()
        if resp.get("ok"):
            logger.info("📨 Telegram notification sent successfully (MarkdownV2)")
        else:
            logger.warning(f"⚠️ MarkdownV2 failed, retrying as plain text. Error: {resp}")
            # Retry without parse_mode
            fallback_payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
            r2 = requests.post(url, json=fallback_payload, timeout=10)
            resp2 = r2.json()
            if resp2.get("ok"):
                logger.info("📨 Telegram notification sent successfully (Plain text fallback)")
            else:
                logger.error(f"❌ Telegram API error even in fallback: {resp2}")
    except Exception as e:
        logger.error(f"❌ Failed to send Telegram message: {e}")
