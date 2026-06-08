# ─────────────────────────────────────────────
#  data_yfinance.py  –  yfinance historical client
#
#  Used by:
#    scoring brains  → daily OHLCV, ATR, pivots
#    signal_report   → historical closes at +1d/+3d/+5d
#    trade_setup     → ATR(14), 60-day pivot high
# ─────────────────────────────────────────────
from __future__ import annotations

import datetime
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    logging.getLogger("data_yfinance").error("yfinance not installed")

from config import REQUEST_DELAY_SECONDS, REQUEST_MAX_RETRIES, REQUEST_RETRY_BACKOFF

logger = logging.getLogger("data_yfinance")


# ── Internal fetch helper ─────────────────────────────────────────────────────

def _yf_history(ticker: str, period: str = "6mo") -> Any:
    """Fetch yfinance history with retry and backoff."""
    if not YF_AVAILABLE:
        return None
    delay = REQUEST_DELAY_SECONDS
    for attempt in range(REQUEST_MAX_RETRIES):
        try:
            time.sleep(delay)
            t    = yf.Ticker(ticker)
            hist = t.history(period=period)
            if hist is not None and not hist.empty:
                return hist
            # Fallback
            if period == "1y":
                return _yf_history(ticker, "6mo")
            return None
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "too many" in msg or "rate" in msg:
                wait = delay * (REQUEST_RETRY_BACKOFF ** attempt)
                logger.warning("yfinance rate limit on %s — waiting %.1fs", ticker, wait)
                time.sleep(wait)
                delay = wait
            else:
                logger.debug("yfinance fetch %s attempt %d: %s", ticker, attempt + 1, e)
                if attempt < REQUEST_MAX_RETRIES - 1:
                    time.sleep(REQUEST_RETRY_BACKOFF ** attempt)
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def get_daily_ohlcv(ticker: str, lookback_days: int = 120) -> List[Dict]:
    """
    Daily OHLCV candles for ATR, pivots, trend analysis.
    Returns list of {date, open, high, low, close, volume}
    """
    hist = _yf_history(ticker, "6mo" if lookback_days <= 126 else "1y")
    if hist is None or hist.empty:
        return []

    candles = []
    for ts, row in hist.tail(lookback_days).iterrows():
        try:
            candles.append({
                "date":   str(ts.date()),
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": float(row["Volume"]),
            })
        except (TypeError, ValueError, KeyError):
            continue
    return candles


def get_close_on_date(ticker: str, target_date: datetime.date) -> Optional[float]:
    """
    Used by signal_report to check +1d/+3d/+5d performance.
    Returns the closing price on or after target_date, or None.
    """
    if not YF_AVAILABLE:
        return None
    try:
        time.sleep(REQUEST_DELAY_SECONDS)
        end  = target_date + datetime.timedelta(days=4)
        hist = yf.Ticker(ticker).history(
            start=target_date.isoformat(),
            end=end.isoformat(),
        )
        if hist is None or hist.empty:
            return None
        return float(hist["Close"].iloc[0])
    except Exception as e:
        logger.debug("get_close_on_date %s %s: %s", ticker, target_date, e)
        return None


def get_current_price(ticker: str) -> Optional[float]:
    """Fast current price via yfinance info (fallback when Webull unavailable)."""
    if not YF_AVAILABLE:
        return None
    try:
        time.sleep(REQUEST_DELAY_SECONDS)
        info = yf.Ticker(ticker).info or {}
        return float(
            info.get("currentPrice") or
            info.get("regularMarketPrice") or
            info.get("previousClose") or 0
        ) or None
    except Exception:
        return None


# ── Technical helpers ─────────────────────────────────────────────────────────

def compute_atr14(candles: List[Dict]) -> float:
    """
    ATR(14) from daily candles.
    True Range = max(H-L, |H-prev_close|, |L-prev_close|)
    """
    if len(candles) < 15:
        closes = [c["close"] for c in candles if c.get("close")]
        return float(np.mean(closes)) * 0.02 if closes else 1.0

    trs = []
    for i in range(1, len(candles)):
        h    = candles[i]["high"]
        l    = candles[i]["low"]
        prev = candles[i - 1]["close"]
        trs.append(max(h - l, abs(h - prev), abs(l - prev)))

    # Use last 14 TRs
    return float(np.mean(trs[-14:]))


def find_60d_pivot_high(candles: List[Dict]) -> float:
    """
    Nearest significant resistance above current price using 60-day pivot highs.
    A pivot high = bar where high > surrounding 3 bars on each side.
    Falls back to recent high if no pivot found above current.
    """
    window = candles[-60:] if len(candles) >= 60 else candles
    if not window:
        return 0.0

    current = window[-1]["close"]
    highs   = [c["high"] for c in window]
    n       = len(highs)

    # Find pivot highs
    pivot_highs = []
    for i in range(3, n - 3):
        if highs[i] == max(highs[i-3:i+4]):
            pivot_highs.append(highs[i])

    # Nearest resistance above current
    above = [p for p in pivot_highs if p > current * 1.001]
    if above:
        return min(above)

    # Fallback: recent 20-bar high
    recent_high = max(highs[-20:]) if len(highs) >= 20 else max(highs)
    return recent_high if recent_high > current else current * 1.03


def compute_support(candles: List[Dict]) -> float:
    """Nearest support below current price using pivot lows."""
    window  = candles[-60:] if len(candles) >= 60 else candles
    if not window:
        return 0.0
    current = window[-1]["close"]
    lows    = [c["low"] for c in window]
    n       = len(lows)

    pivot_lows = []
    for i in range(3, n - 3):
        if lows[i] == min(lows[i-3:i+4]):
            pivot_lows.append(lows[i])

    below = [p for p in pivot_lows if p < current * 0.999]
    return max(below) if below else current * 0.93


def compute_volume_avg(candles: List[Dict], days: int = 20) -> float:
    """Average volume over last N days."""
    vols = [c["volume"] for c in candles[-days:] if c.get("volume")]
    return float(np.mean(vols)) if vols else 0.0


def compute_rsi(candles: List[Dict], period: int = 14) -> float:
    """RSI(14) from closing prices."""
    closes = np.array([c["close"] for c in candles if c.get("close")])
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = float(np.mean(gains))
    avg_l  = float(np.mean(losses))
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return round(100.0 - 100.0 / (1.0 + rs), 2)
