# ─────────────────────────────────────────────
#  alerts/notifier.py  –  Alert dispatcher
# ─────────────────────────────────────────────
from __future__ import annotations
import json
import os
import datetime
from typing import Any, Dict

from config import NOTIFICATION_CONFIG


def _format_alert(payload: Dict[str, Any]) -> str:
    """Human-readable alert string."""
    ticker   = payload["ticker"]
    upside   = payload["upside"]
    risk     = payload["risk"]
    regime   = payload["regime"]
    setups   = payload.get("setups", [])
    factors  = payload.get("factor_scores", {})
    issues   = payload.get("issues", [])
    change   = payload.get("upside_change", 0.0)

    lines = [
        "=" * 55,
        f"  📈 ALERT  →  {ticker}",
        "=" * 55,
        f"  UpsideScore : {upside:.3f}  (+{change:.3f} change)",
        f"  RiskScore   : {risk:.3f}",
        f"  Regime      : {regime}",
        f"  Setups      : {', '.join(setups) if setups else 'none'}",
        "",
        "  Factor breakdown:",
    ]
    for k, v in factors.items():
        bar = "█" * int(v * 15) + "░" * (15 - int(v * 15))
        lines.append(f"    {k:12s}  [{bar}]  {v:.2f}")

    if issues:
        lines.append(f"\n  ⚠  Warnings: {'; '.join(issues)}")

    lines.append("=" * 55)
    return "\n".join(lines)


def send_alert(alert_payload: Dict[str, Any]) -> None:
    """
    Dispatch an alert according to NOTIFICATION_CONFIG["channel"].

    Supported channels: "console", "file"
    Stub channels: "telegram", "email" (see TODO comments)
    """
    channel = NOTIFICATION_CONFIG.get("channel", "console")

    if channel == "console":
        print(_format_alert(alert_payload))

    elif channel == "file":
        path = NOTIFICATION_CONFIG.get("output_file", "alerts.jsonl")
        record = dict(alert_payload)
        record["timestamp"] = datetime.datetime.utcnow().isoformat()
        # Convert numpy arrays to plain lists if present
        for k, v in record.items():
            if hasattr(v, "tolist"):
                record[k] = v.tolist()
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[notifier] Alert for {alert_payload['ticker']} written to {path}")

    elif channel == "telegram":
        # TODO: implement
        # import requests
        # token = os.environ["TELEGRAM_BOT_TOKEN"]
        # chat  = os.environ["TELEGRAM_CHAT_ID"]
        # msg   = _format_alert(alert_payload)
        # requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
        #               json={"chat_id": chat, "text": msg})
        print(f"[notifier] Telegram stub – alert for {alert_payload['ticker']}")

    elif channel == "email":
        # TODO: implement SMTP
        print(f"[notifier] Email stub – alert for {alert_payload['ticker']}")

    else:
        print(f"[notifier] Unknown channel '{channel}' – printing to console")
        print(_format_alert(alert_payload))
