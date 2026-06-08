# ─────────────────────────────────────────────
#  config.py  –  Global configuration
#  Layered architecture:
#    yfinance  → scoring + historical
#    Webull    → real-time quotes + trading
# ─────────────────────────────────────────────
import os

# ── Webull Open API ───────────────────────────────────────────────────────────
WEBULL_APP_KEY      = os.getenv("WEBULL_APP_KEY",      "")
WEBULL_APP_SECRET   = os.getenv("WEBULL_APP_SECRET",   "")
WEBULL_ENABLED      = bool(WEBULL_APP_KEY and WEBULL_APP_SECRET)
WEBULL_PAPER_TRADING = True   # set False for live trading

# ── Trading gate ──────────────────────────────────────────────────────────────
# Hard gate: no orders sent unless explicitly set True
LIVE_TRADING_ENABLED = os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"

# ── Notifications ─────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID",   "")
EMAIL_ENABLED      = os.environ.get("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_SMTP_HOST    = os.environ.get("EMAIL_SMTP_HOST",  "smtp.gmail.com")
EMAIL_SMTP_PORT    = int(os.environ.get("EMAIL_SMTP_PORT", "587"))
EMAIL_USERNAME     = os.environ.get("EMAIL_USERNAME",  "")
EMAIL_PASSWORD     = os.environ.get("EMAIL_PASSWORD",  "")
EMAIL_TO           = os.environ.get("EMAIL_TO",        "")
FINNHUB_API_KEY    = os.environ.get("FINNHUB_API_KEY", "")

# ── Universe definitions ──────────────────────────────────────────────────────
UNIVERSE_CONFIG = {
    "min_price":             2.0,
    "min_avg_dollar_volume": 1_000_000,
    "include_small_caps":    True,
    "include_micro_caps":    False,
    "ticker_list":           None,
    "mode":                  "sp500_top100",
}

UNIVERSE = {
    "sp_top_100":        [],   # populated dynamically by universe.py
    "under_10":          [],
    "under_5":           [],
    "low_float_rockets": [],
    "short_squeeze":     [],
    "penny_radar":       [],
    "pre_earnings":      [],
}

# ── Section refresh intervals (API fetch cadence, not UI) ─────────────────────
REFRESH_INTERVALS = {
    "top_alerts":        5,
    "low_float_rockets": 5,
    "volume_anomalies":  5,
    "under_5":           5,
    "penny_radar":       5,
    "under_10":          10,
    "short_squeeze":     10,
    "squeeze_box":       15,
    "pre_earnings":      30,
}

# ── yfinance settings ─────────────────────────────────────────────────────────
YFINANCE_ENABLED       = True
REQUEST_DELAY_SECONDS  = 0.35
REQUEST_MAX_RETRIES    = 3
REQUEST_RETRY_BACKOFF  = 2.0

# ── Alert thresholds ──────────────────────────────────────────────────────────
ALERT_CONFIG = {
    "upside_percentile_threshold": 0.60,
    "risk_percentile_max":         0.65,
    "setup_percentile_threshold":  0.55,
    "min_upside_change":           0.02,
    "min_regime_score":            0.40,
}

# ── History ───────────────────────────────────────────────────────────────────
HISTORY_CONFIG = {
    "lookback_days":       252,
    "factor_history_days":  60,
    "regime_history_days":  90,
    "persistence_file":    "history_store.json",
    "alert_log_file":      "alert_history.jsonl",
    "max_history_per_key":  90,
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

# ── Scan timing ───────────────────────────────────────────────────────────────
SCAN_INTERVAL_SECONDS = 600

# ── Position sizing ───────────────────────────────────────────────────────────
POSITION_SIZING = {
    "account_size":       25_000,
    "risk_per_trade_pct":  0.01,
    "max_position_pct":    0.10,
}

# ── Options ───────────────────────────────────────────────────────────────────
OPTIONS_CONFIG = {
    "low_iv_threshold":   30,
    "high_iv_threshold":  50,
    "min_days_to_expiry": 14,
    "max_days_to_expiry": 60,
    "max_spread_width":   10,
    "min_open_interest":  100,
}

# ── Misc ──────────────────────────────────────────────────────────────────────
PRICE_ALERTS_FILE = "price_alerts.json"
NOTIFICATION_CONFIG = {
    "channel":          "console",
    "output_file":      "alerts.jsonl",
    "telegram_enabled": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
    "email_enabled":    EMAIL_ENABLED,
}
SECTOR_LIST = [
    "Technology", "Healthcare", "Financials", "Consumer Cyclical",
    "Consumer Defensive", "Energy", "Industrials", "Materials",
    "Utilities", "Real Estate", "Communication Services",
]
DROP_THRESHOLD     = 0.10
DROP_LOOKBACK_DAYS = 5
