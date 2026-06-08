# ─────────────────────────────────────────────
#  config.py  –  Global configuration
# ─────────────────────────────────────────────
import os

# ── External API keys ─────────────────────────────────────────────────────────
FINNHUB_API_KEY    = os.environ.get("FINNHUB_API_KEY",    "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID",   "")
EMAIL_ENABLED      = os.environ.get("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_SMTP_HOST    = os.environ.get("EMAIL_SMTP_HOST",  "smtp.gmail.com")
EMAIL_SMTP_PORT    = int(os.environ.get("EMAIL_SMTP_PORT", "587"))
EMAIL_USERNAME     = os.environ.get("EMAIL_USERNAME",  "")
EMAIL_PASSWORD     = os.environ.get("EMAIL_PASSWORD",  "")
EMAIL_TO           = os.environ.get("EMAIL_TO",        "")

# ── Universe ──────────────────────────────────────────────────────────────────
DROP_THRESHOLD     = 0.10
DROP_LOOKBACK_DAYS = 5

UNIVERSE_CONFIG = {
    "min_price":             2.0,
    "min_avg_dollar_volume": 1_000_000,
    "include_small_caps":    True,
    "include_micro_caps":    False,
    "ticker_list":           None,
    # "sp500_full" | "sp500_top100" | "small_caps" | "under20" |
    # "high_short_interest"  | "watchlist"
    "mode": "sp500_top100",
}

# ── Alert thresholds ──────────────────────────────────────────────────────────
ALERT_CONFIG = {
    "upside_percentile_threshold": 0.65,
    "risk_percentile_max":         0.60,
    "setup_percentile_threshold":  0.55,
    "min_upside_change":           0.05,
    "min_regime_score":            0.40,
}

# ── History ───────────────────────────────────────────────────────────────────
HISTORY_CONFIG = {
    "lookback_days":      252,
    "factor_history_days": 60,
    "regime_history_days": 90,
    "persistence_file":   "history_store.json",
    "alert_log_file":     "alert_history.jsonl",
    "max_history_per_key": 90,   # keep last N data points per ticker/key
}

# ── Notifications ─────────────────────────────────────────────────────────────
NOTIFICATION_CONFIG = {
    "channel":          "console",
    "output_file":      "alerts.jsonl",
    "telegram_enabled": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
    "email_enabled":    EMAIL_ENABLED,
}

# ── Scoring ───────────────────────────────────────────────────────────────────
SCORING_WEIGHTS = {
    "technical":   0.25,
    "fundamental": 0.25,
    "sentiment":   0.15,
    "structural":  0.15,
    "setup":       0.10,
    "regime":      0.10,
}
ADAPTIVE_WEIGHT_WINDOW  = 30
ADAPTIVE_WEIGHT_ENABLED = True

# ── Scan / request timing ─────────────────────────────────────────────────────
SCAN_INTERVAL_SECONDS  = 600
REQUEST_DELAY_SECONDS  = 0.35
REQUEST_MAX_RETRIES    = 3
REQUEST_RETRY_BACKOFF  = 2.0

# ── Position sizing ───────────────────────────────────────────────────────────
POSITION_SIZING = {
    "account_size":       25_000,
    "risk_per_trade_pct":  0.01,
    "max_position_pct":    0.10,
}

# ── Price alerts ──────────────────────────────────────────────────────────────
PRICE_ALERTS_FILE = "price_alerts.json"

# ── Options plays ─────────────────────────────────────────────────────────────
OPTIONS_CONFIG = {
    "low_iv_threshold":   30,
    "high_iv_threshold":  50,
    "min_days_to_expiry": 14,
    "max_days_to_expiry": 60,
    "max_spread_width":   10,
    "min_open_interest":  100,
}

# ── Sectors ───────────────────────────────────────────────────────────────────
SECTOR_LIST = [
    "Technology", "Healthcare", "Financials", "Consumer Cyclical",
    "Consumer Defensive", "Energy", "Industrials", "Materials",
    "Utilities", "Real Estate", "Communication Services",
]
