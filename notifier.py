# ─────────────────────────────────────────────
#  notifier.py  –  Alert dispatcher
#  Supports console, file, Telegram, email
# ─────────────────────────────────────────────
from __future__ import annotations
import datetime
import json
import logging
import os
import smtplib
from email.mime.text import MIMEText
from typing import Any, Dict

from config import NOTIFICATION_CONFIG, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from config import EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, EMAIL_USERNAME, EMAIL_PASSWORD, EMAIL_TO

logger = logging.getLogger("notifier")


def _format_alert(payload: Dict[str, Any]) -> str:
    ticker  = payload["ticker"]
    upside  = payload["upside"]
    risk    = payload["risk"]
    regime  = payload["regime"]
    setups  = payload.get("setups", [])
    factors = payload.get("factor_scores", {})
    change  = payload.get("upside_change", 0.0)
    price   = payload.get("last_price", 0)
    is_new  = payload.get("is_new", False)

    lines = [
        "=" * 55,
        f"  {'🆕 NEW ' if is_new else ''}📈 ALERT  →  {ticker}  ${price:.2f}",
        "=" * 55,
        f"  UpsideScore : {upside:.3f}  (+{change:.3f})",
        f"  RiskScore   : {risk:.3f}",
        f"  Regime      : {regime}",
        f"  Setups      : {', '.join(setups) if setups else 'none'}",
        "",
        "  Factor breakdown:",
    ]
    for k, v in factors.items():
        bar = "█" * int(v * 15) + "░" * (15 - int(v * 15))
        lines.append(f"    {k:12s}  [{bar}]  {v:.2f}")
    lines.append("=" * 55)
    return "\n".join(lines)


def _send_telegram(message: str):
    try:
        import requests
        url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       message,
            "parse_mode": "HTML",
        }, timeout=10)
        resp.raise_for_status()
        logger.info("Telegram alert sent")
    except Exception as e:
        logger.warning("Telegram send failed: %s", e)


def _send_email(subject: str, body: str):
    try:
        msg            = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = EMAIL_USERNAME
        msg["To"]      = EMAIL_TO
        with smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info("Email alert sent to %s", EMAIL_TO)
    except Exception as e:
        logger.warning("Email send failed: %s", e)


def send_alert(alert_payload: Dict[str, Any]) -> None:
    channel  = NOTIFICATION_CONFIG.get("channel", "console")
    tg_on    = NOTIFICATION_CONFIG.get("telegram_enabled", False)
    email_on = NOTIFICATION_CONFIG.get("email_enabled",    False)
    message  = _format_alert(alert_payload)
    ticker   = alert_payload.get("ticker", "?")

    if channel in ("console", "all"):
        print(message)

    if channel in ("file", "all"):
        path   = NOTIFICATION_CONFIG.get("output_file", "alerts.jsonl")
        record = dict(alert_payload)
        record["timestamp"] = datetime.datetime.utcnow().isoformat()
        for k, v in record.items():
            if hasattr(v, "tolist"):
                record[k] = v.tolist()
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    if (channel in ("telegram", "all") or tg_on) and TELEGRAM_BOT_TOKEN:
        _send_telegram(message)

    if (channel in ("email", "all") or email_on) and EMAIL_USERNAME:
        _send_email(f"StockBot Alert: {ticker}", message)
