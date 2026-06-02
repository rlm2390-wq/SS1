# ─────────────────────────────────────────────
#  config.py  –  Global configuration
# ─────────────────────────────────────────────
import os

DATA_PROVIDER = "synthetic"   # swap for "alpaca", "polygon", etc.

# ── Universe / IPO / drop detection ───────────────────────────────────────────
# Optional: set FINNHUB_API_KEY to enable IPO calendar detection.
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

# Tickers down this fraction (0.10 = 10%) over DROP_LOOKBACK_DAYS are included
# in the scan universe as "interesting" candidates.
DROP_THRESHOLD    = 0.10   # 10%
DROP_LOOKBACK_DAYS = 5

UNIVERSE_CONFIG = {
    "min_price": 2.0,
    "min_avg_dollar_volume": 1_000_000,
    "include_small_caps": True,
    "include_micro_caps": False,
    # Set to None to use synthetic demo tickers
    "ticker_list": None,
}

ALERT_CONFIG = {
    # Upside must be above this absolute value (0–1)
    "upside_percentile_threshold": 0.65,
    # Risk must be below this
    "risk_percentile_max": 0.60,
    # Setup score must be above this
    "setup_percentile_threshold": 0.55,
    # Minimum absolute improvement in UpsideScore vs previous run
    "min_upside_change": 0.05,
    # Minimum regime score to fire any alert
    "min_regime_score": 0.40,
}

HISTORY_CONFIG = {
    "lookback_days": 252,
    "factor_history_days": 60,
    "regime_history_days": 90,
}

NOTIFICATION_CONFIG = {
    # Options: "console" | "file" | "telegram" | "email"
    "channel": "console",
    "output_file": "alerts.jsonl",
    # Telegram: set BOT_TOKEN and CHAT_ID env vars if using telegram
    # Email: set SMTP_* env vars if using email
}

SCORING_WEIGHTS = {
    # Static fallback weights (adaptive weighting overrides these)
    "technical":   0.25,
    "fundamental": 0.25,
    "sentiment":   0.15,
    "structural":  0.15,
    "setup":       0.10,
    "regime":      0.10,
}

# Adaptive weighting: rolling window (days) for correlation-based weight updates
ADAPTIVE_WEIGHT_WINDOW = 30
ADAPTIVE_WEIGHT_ENABLED = True
