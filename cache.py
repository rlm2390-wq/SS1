# ─────────────────────────────────────────────
#  cache.py  –  Unified cache layer
#
#  Thread-safe in-memory cache with per-key TTLs.
#  Dashboard reads from cache; never hits API directly.
#
#  TTLs:
#    real-time quotes:   5s
#    intraday candles:  30s
#    daily OHLCV:       10m
#    ATR / pivots:      10m
#    S&P universe:      24h
# ─────────────────────────────────────────────
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

_lock  = threading.RLock()
_store: Dict[str, Tuple[Any, float]] = {}   # key → (value, expires_at)

# ── TTL constants (seconds) ───────────────────────────────────────────────────
TTL_REALTIME  =     5
TTL_INTRADAY  =    30
TTL_DAILY     =   600   # 10 minutes
TTL_UNIVERSE  = 86400   # 24 hours
TTL_PREMARKET =    30


def set(key: str, value: Any, ttl: float = TTL_DAILY) -> None:
    with _lock:
        _store[key] = (value, time.monotonic() + ttl)


def get(key: str) -> Optional[Any]:
    with _lock:
        entry = _store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del _store[key]
            return None
        return value


def get_or_set(key: str, fn: Callable[[], Any], ttl: float = TTL_DAILY) -> Any:
    """Return cached value, or call fn() to populate and cache it."""
    v = get(key)
    if v is not None:
        return v
    v = fn()
    if v is not None:
        set(key, v, ttl)
    return v


def invalidate(key: str) -> None:
    with _lock:
        _store.pop(key, None)


def invalidate_prefix(prefix: str) -> int:
    """Remove all keys starting with prefix. Returns count removed."""
    with _lock:
        keys = [k for k in _store if k.startswith(prefix)]
        for k in keys:
            del _store[k]
        return len(keys)


def ttl_remaining(key: str) -> Optional[float]:
    with _lock:
        entry = _store.get(key)
        if entry is None:
            return None
        _, expires_at = entry
        remaining = expires_at - time.monotonic()
        return max(0.0, remaining)


def stats() -> Dict[str, int]:
    now = time.monotonic()
    with _lock:
        total   = len(_store)
        expired = sum(1 for _, (_, exp) in _store.items() if now > exp)
    return {"total_keys": total, "expired_keys": expired, "live_keys": total - expired}


# ── Convenience wrappers for common cache patterns ────────────────────────────

def cache_quote(ticker: str, data: Dict) -> None:
    set(f"quote:{ticker}", data, TTL_REALTIME)

def get_quote(ticker: str) -> Optional[Dict]:
    return get(f"quote:{ticker}")

def cache_quotes_batch(quotes: Dict[str, Dict]) -> None:
    for ticker, data in quotes.items():
        cache_quote(ticker, data)

def cache_intraday(ticker: str, candles: list) -> None:
    set(f"intraday:{ticker}", candles, TTL_INTRADAY)

def get_intraday(ticker: str) -> Optional[list]:
    return get(f"intraday:{ticker}")

def cache_daily(ticker: str, candles: list) -> None:
    set(f"daily:{ticker}", candles, TTL_DAILY)

def get_daily(ticker: str) -> Optional[list]:
    return get(f"daily:{ticker}")

def cache_atr(ticker: str, atr: float) -> None:
    set(f"atr:{ticker}", atr, TTL_DAILY)

def get_atr(ticker: str) -> Optional[float]:
    return get(f"atr:{ticker}")

def cache_pivot(ticker: str, pivot: float) -> None:
    set(f"pivot:{ticker}", pivot, TTL_DAILY)

def get_pivot(ticker: str) -> Optional[float]:
    return get(f"pivot:{ticker}")

def cache_universe(mode: str, tickers: list) -> None:
    set(f"universe:{mode}", tickers, TTL_UNIVERSE)

def get_universe_cached(mode: str) -> Optional[list]:
    return get(f"universe:{mode}")

def cache_premarket(data: Dict) -> None:
    set("premarket:latest", data, TTL_PREMARKET)

def get_premarket_cached() -> Optional[Dict]:
    return get("premarket:latest")
