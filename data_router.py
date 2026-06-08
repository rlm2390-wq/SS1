# ─────────────────────────────────────────────
#  data_router.py  –  Unified data source router
#
#  Decides which backend provides each data type:
#    yfinance  → historical OHLCV, scoring, ATR, pivots
#    Webull    → real-time quotes, intraday, options
#
#  Currently REALTIME_ENABLED = False (yfinance fallback).
#  Switching to Webull is a one-line config change.
#
#  All consumers call data_router — never yfinance/Webull directly.
# ─────────────────────────────────────────────
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cache

logger = logging.getLogger("data_router")

# ── Feature flag ──────────────────────────────────────────────────────────────
# Set REALTIME_ENABLED = True once Webull credentials are configured.
# Everything else is automatic.
import os
REALTIME_ENABLED: bool = (
    os.getenv("WEBULL_APP_KEY", "") != "" and
    os.getenv("WEBULL_APP_SECRET", "") != ""
)


# ── Quote routing ─────────────────────────────────────────────────────────────

def get_quote(ticker: str) -> Optional[Dict]:
    """
    Real-time or near-real-time quote for a single ticker.
    Returns cached value first; fetches if stale.
    """
    cached = cache.get_quote(ticker)
    if cached:
        return cached

    if REALTIME_ENABLED:
        try:
            from data_webull import get_client
            q = get_client().get_quote(ticker)
            if q and q.get("available"):
                cache.cache_quote(ticker, q)
                return q
        except Exception as e:
            logger.debug("Webull quote failed for %s: %s", ticker, e)

    # yfinance fallback
    try:
        from data_yfinance import get_current_price
        price = get_current_price(ticker)
        if price:
            q = {"ticker": ticker, "last": price, "available": True, "source": "yfinance"}
            cache.cache_quote(ticker, q)
            return q
    except Exception as e:
        logger.debug("yfinance quote failed for %s: %s", ticker, e)

    return None


def get_quotes_batch(tickers: List[str]) -> Dict[str, Dict]:
    """
    Batch quotes. Uses Webull batch endpoint if available,
    else falls back to cached yfinance prices from last scan.
    """
    results: Dict[str, Dict] = {}

    # Check cache first
    uncached = []
    for t in tickers:
        cached = cache.get_quote(t)
        if cached:
            results[t] = cached
        else:
            uncached.append(t)

    if not uncached:
        return results

    if REALTIME_ENABLED:
        try:
            from data_webull import get_client
            fresh = get_client().get_quotes_batch(uncached)
            cache.cache_quotes_batch(fresh)
            results.update(fresh)
            return results
        except Exception as e:
            logger.debug("Webull batch quotes failed: %s", e)

    # yfinance fallback — use prices already in scan results
    # (no extra API calls — quotes are populated by the scan loop)
    return results


# ── Intraday candles ──────────────────────────────────────────────────────────

def get_intraday_candles(
    ticker: str,
    interval: str = "m1",
    lookback_minutes: int = 60,
) -> List[Dict]:
    """
    Intraday candles. Webull if available, else empty (yfinance has no intraday).
    """
    cached = cache.get_intraday(ticker)
    if cached is not None:
        return cached

    if REALTIME_ENABLED:
        try:
            from data_webull import get_client
            candles = get_client().get_intraday_candles(ticker, interval, lookback_minutes)
            cache.cache_intraday(ticker, candles)
            return candles
        except Exception as e:
            logger.debug("Webull intraday failed for %s: %s", ticker, e)

    return []


# ── Daily OHLCV ───────────────────────────────────────────────────────────────

def get_daily_ohlcv(ticker: str, lookback_days: int = 120) -> List[Dict]:
    """
    Daily OHLCV — always yfinance (historical source).
    Cached for 10 minutes to avoid redundant fetches during scans.
    """
    cached = cache.get_daily(ticker)
    if cached is not None:
        return cached

    from data_yfinance import get_daily_ohlcv as _get
    candles = _get(ticker, lookback_days)
    if candles:
        cache.cache_daily(ticker, candles)
    return candles


# ── ATR ───────────────────────────────────────────────────────────────────────

def get_atr14(ticker: str, candles: Optional[List[Dict]] = None) -> float:
    """ATR(14) — cached for 10 minutes."""
    cached = cache.get_atr(ticker)
    if cached is not None:
        return cached

    if candles is None:
        candles = get_daily_ohlcv(ticker)

    from data_yfinance import compute_atr14
    atr = compute_atr14(candles) if candles else 0.02
    cache.cache_atr(ticker, atr)
    return atr


# ── Pivot high ────────────────────────────────────────────────────────────────

def get_pivot_high(ticker: str, candles: Optional[List[Dict]] = None) -> float:
    """60-day pivot high — cached for 10 minutes."""
    cached = cache.get_pivot(ticker)
    if cached is not None:
        return cached

    if candles is None:
        candles = get_daily_ohlcv(ticker)

    from data_yfinance import find_60d_pivot_high
    pivot = find_60d_pivot_high(candles) if candles else 0.0
    cache.cache_pivot(ticker, pivot)
    return pivot


# ── Options chain ─────────────────────────────────────────────────────────────

def get_options_chain(ticker: str) -> Optional[Dict]:
    """Options chain — Webull if available, else None (yfinance options are unreliable)."""
    if REALTIME_ENABLED:
        try:
            from data_webull import get_client
            return get_client().get_options_chain(ticker)
        except Exception as e:
            logger.debug("Webull options failed for %s: %s", ticker, e)
    return None


# ── Status ────────────────────────────────────────────────────────────────────

def status() -> Dict[str, Any]:
    return {
        "realtime_enabled": REALTIME_ENABLED,
        "source_quotes":    "webull" if REALTIME_ENABLED else "yfinance",
        "source_historical": "yfinance",
        "cache_stats":      cache.stats(),
    }
