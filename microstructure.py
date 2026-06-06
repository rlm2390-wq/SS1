# ─────────────────────────────────────────────
#  microstructure.py  –  Market Microstructure Brain
#
#  Institutional-grade tape-reading intelligence
#  using intraday (1-day interval) OHLCV data.
#
#  Signals:
#    1. Dollar volume acceleration  — institutions move $, not shares
#    2. Price efficiency ratio      — trending cleanly vs choppy/noisy
#    3. Bid-ask spread proxy        — intraday range compression/expansion
#    4. Accumulation/distribution   — up-vol vs down-vol balance
#    5. Gap analysis                — overnight bidding pattern
#    6. Price impact sensitivity    — supply absorption detection
#    7. VWAP drift                  — price vs volume-weighted avg
#    8. Intraday wick analysis      — rejection vs acceptance at levels
#
#  Output: microstructure_score 0-1
#    > 0.65 = institutional accumulation pattern
#    < 0.35 = distribution / weak tape
# ─────────────────────────────────────────────
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple

import numpy as np

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

from history import HistoryStore
from config import REQUEST_DELAY_SECONDS, REQUEST_MAX_RETRIES, REQUEST_RETRY_BACKOFF

logger = logging.getLogger("microstructure")


# ── Intraday OHLCV fetch with caching ─────────────────────────────────────────

_intraday_cache: Dict[str, Tuple[float, Any]] = {}
_CACHE_TTL = 3600   # 1 hour


def _fetch_intraday(ticker: str) -> Any:
    """
    Fetch 1-month of 1-day OHLCV data.
    Returns a dict with arrays: open, high, low, close, volume.
    Cached per ticker for 1 hour.
    """
    now = time.time()
    if ticker in _intraday_cache:
        ts, cached = _intraday_cache[ticker]
        if now - ts < _CACHE_TTL:
            return cached

    if not YF_AVAILABLE:
        return None

    delay = REQUEST_DELAY_SECONDS
    for attempt in range(REQUEST_MAX_RETRIES):
        try:
            time.sleep(delay)
            t    = yf.Ticker(ticker)
            hist = t.history(period="2mo", interval="1d")
            if hist.empty or len(hist) < 10:
                return None

            data = {
                "open":   hist["Open"].values.astype(float),
                "high":   hist["High"].values.astype(float),
                "low":    hist["Low"].values.astype(float),
                "close":  hist["Close"].values.astype(float),
                "volume": hist["Volume"].values.astype(float),
            }
            _intraday_cache[ticker] = (now, data)
            return data

        except Exception as e:
            msg = str(e).lower()
            if "too many" in msg or "429" in msg or "rate" in msg:
                wait = delay * (REQUEST_RETRY_BACKOFF ** attempt)
                logger.debug("Rate limit on %s attempt %d, sleeping %.1fs", ticker, attempt+1, wait)
                time.sleep(wait)
                delay = wait
            else:
                logger.debug("Intraday fetch failed for %s: %s", ticker, e)
                return None

    return None


# ── Signal functions ──────────────────────────────────────────────────────────

def _dollar_vol_acceleration(d: Dict) -> float:
    """
    Dollar volume (price × volume) acceleration over last 5 days
    vs 20-day baseline. Higher = institutional activity picking up.
    Returns 0–1 score.
    """
    close   = d["close"]
    volume  = d["volume"]
    dv      = close * volume

    if len(dv) < 20:
        return 0.5

    dv_recent  = float(dv[-5:].mean())
    dv_baseline= float(dv[-20:].mean())

    ratio = dv_recent / max(dv_baseline, 1e-9)
    # 2x = very elevated, 0.5x = very low
    if   ratio > 2.0:  return 0.95
    elif ratio > 1.5:  return 0.80
    elif ratio > 1.2:  return 0.65
    elif ratio > 0.8:  return 0.50
    elif ratio > 0.5:  return 0.30
    else:              return 0.15


def _price_efficiency_ratio(d: Dict) -> float:
    """
    Ratio of net directional move to total path traveled over last 10 days.
    High efficiency (close to 1.0) = trending cleanly.
    Low efficiency (close to 0) = choppy/noisy.
    """
    close = d["close"]
    if len(close) < 10:
        return 0.5

    window    = close[-10:]
    net_move  = abs(float(window[-1]) - float(window[0]))
    total_path= float(np.sum(np.abs(np.diff(window))))

    if total_path < 1e-9:
        return 0.5

    ratio = net_move / total_path
    return float(np.clip(ratio, 0.0, 1.0))


def _spread_proxy(d: Dict) -> float:
    """
    Bid-ask spread proxy: (high - low) / close vs 20-day average.
    Narrowing spreads = institutions stepping in confidently.
    Widening spreads = uncertainty or distribution.
    Returns 0–1 (higher = tighter/better spreads).
    """
    high  = d["high"]
    low   = d["low"]
    close = d["close"]

    if len(close) < 10:
        return 0.5

    spread     = (high - low) / (close + 1e-9)
    recent_sp  = float(spread[-5:].mean())
    baseline_sp= float(spread[-20:].mean()) if len(spread) >= 20 else recent_sp

    ratio = recent_sp / max(baseline_sp, 1e-9)
    # Narrowing = good (score > 0.5), widening = bad (score < 0.5)
    score = 1.0 - float(np.clip((ratio - 0.5) / 1.5, 0.0, 1.0))
    return float(np.clip(score, 0.0, 1.0))


def _accumulation_distribution(d: Dict) -> float:
    """
    Up-day dollar volume vs down-day dollar volume over last 20 days.
    Positive asymmetry = net accumulation.
    Returns 0–1 (0.5 = neutral).
    """
    close  = d["close"]
    volume = d["volume"]

    if len(close) < 5:
        return 0.5

    diffs    = np.diff(close[-21:])
    vols_20  = volume[-20:]

    if len(diffs) != len(vols_20):
        n = min(len(diffs), len(vols_20))
        diffs    = diffs[-n:]
        vols_20  = vols_20[-n:]

    dv       = (close[-len(vols_20):]) * vols_20
    up_dv    = float(dv[diffs > 0].sum())
    down_dv  = float(dv[diffs < 0].sum())
    total_dv = up_dv + down_dv

    if total_dv < 1e-9:
        return 0.5

    return float(np.clip(up_dv / total_dv, 0.0, 1.0))


def _gap_analysis(d: Dict) -> float:
    """
    Frequency and direction of overnight gaps over last 30 days.
    More up-gaps = stock is being bid after hours (institutional orders).
    Returns 0–1.
    """
    open_  = d["open"]
    close_ = d["close"]

    if len(open_) < 5:
        return 0.5

    n       = min(len(open_), len(close_), 30)
    opens   = open_[-n:]
    closes  = close_[-n-1:-1] if len(close_) > n else close_

    if len(closes) == 0:
        return 0.5

    gaps    = opens[:len(closes)] - closes
    sig_gap = 0.005  # 0.5% threshold for a meaningful gap

    up_gaps   = float(np.sum(gaps >  sig_gap))
    down_gaps = float(np.sum(gaps < -sig_gap))

    if up_gaps + down_gaps < 3:
        return 0.5

    up_ratio = up_gaps / (up_gaps + down_gaps)
    return float(np.clip(up_ratio, 0.0, 1.0))


def _price_impact(d: Dict) -> float:
    """
    Price move per unit of dollar volume.
    Declining price impact = large player absorbing supply (stealth accumulation).
    Increasing price impact = liquidity thinning (squeeze or breakout imminent).

    Returns 0–1:
      > 0.65 = declining impact (absorption)
      0.35–0.65 = neutral
      < 0.35 = increasing impact (could mean thinning liquidity)
    """
    close  = d["close"]
    volume = d["volume"]

    if len(close) < 15:
        return 0.5

    dv           = close * volume
    price_moves  = np.abs(np.diff(close))
    dv_mid       = dv[:-1]

    if len(dv_mid) < 10:
        return 0.5

    # Avoid division by near-zero
    valid = dv_mid > 1e-9
    if valid.sum() < 5:
        return 0.5

    impact = price_moves[valid] / dv_mid[valid]

    recent_impact   = float(np.mean(impact[-5:]))
    baseline_impact = float(np.mean(impact[-15:]))

    ratio = recent_impact / max(baseline_impact, 1e-12)

    # Declining impact = absorption = bullish signal
    if   ratio < 0.7:  return 0.85
    elif ratio < 0.9:  return 0.65
    elif ratio < 1.1:  return 0.50
    elif ratio < 1.3:  return 0.35
    else:              return 0.20


def _vwap_drift(d: Dict) -> float:
    """
    Compare closing price to a 5-day VWAP.
    Price consistently closing above VWAP = buyers in control.
    Returns 0–1.
    """
    close  = d["close"]
    volume = d["volume"]
    high   = d["high"]
    low    = d["low"]

    if len(close) < 5:
        return 0.5

    # Typical price VWAP
    typical = (high + low + close) / 3
    n       = min(len(typical), 10)
    vwap_vals = []
    for i in range(n):
        idx = len(typical) - n + i
        if idx < 0:
            continue
        tp_slice = typical[max(0, idx-4):idx+1]
        vl_slice = volume[max(0, idx-4):idx+1]
        if vl_slice.sum() > 0:
            vwap_vals.append(float(np.average(tp_slice, weights=vl_slice)))

    if not vwap_vals:
        return 0.5

    last_vwap  = vwap_vals[-1]
    last_close = float(close[-1])

    if last_vwap <= 0:
        return 0.5

    drift = (last_close - last_vwap) / last_vwap
    return float(np.clip(0.5 + drift / 0.03, 0.0, 1.0))


def _wick_analysis(d: Dict) -> float:
    """
    Analyze wick patterns over last 10 days.
    Long lower wicks (rejection of lows) = buyers absorbing sellers.
    Long upper wicks (rejection of highs) = distribution.
    Returns 0–1 (higher = bullish wick pattern).
    """
    open_  = d["open"]
    high   = d["high"]
    low    = d["low"]
    close  = d["close"]

    if len(close) < 5:
        return 0.5

    n      = min(10, len(close))
    o, h, l, c = open_[-n:], high[-n:], low[-n:], close[-n:]

    body       = np.abs(c - o)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    total_range= h - l + 1e-9

    # Bullish: lower wicks long, upper wicks short
    lower_ratio = float(np.mean(lower_wick / total_range))
    upper_ratio = float(np.mean(upper_wick / total_range))
    wick_bias   = lower_ratio - upper_ratio   # positive = bullish

    return float(np.clip(0.5 + wick_bias * 2.5, 0.0, 1.0))


# ── Public API ────────────────────────────────────────────────────────────────

_WEIGHTS = {
    "dollar_vol_accel":   0.20,
    "price_efficiency":   0.15,
    "spread_proxy":       0.10,
    "accum_dist":         0.20,
    "gap_analysis":       0.10,
    "price_impact":       0.10,
    "vwap_drift":         0.10,
    "wick_analysis":      0.05,
}


def compute_microstructure(
    ticker: str,
    stock_data: Dict[str, Any],
    history_store: HistoryStore,
) -> Dict[str, Any]:
    """
    Compute the microstructure score for a ticker.

    Returns:
    {
        "microstructure_score": float 0-1,
        "sub_scores":           dict of signal name → score,
        "interpretation":       str human-readable summary,
        "data_quality":         "full" | "partial" | "fallback",
    }
    """
    # Try to fetch intraday OHLCV for richer signals
    intraday = _fetch_intraday(ticker)

    if intraday is not None and len(intraday["close"]) >= 10:
        d            = intraday
        data_quality = "full"
    else:
        # Fall back to daily data from stock_data
        prices  = stock_data.get("prices",  np.array([]))
        volumes = stock_data.get("volumes", np.array([]))
        if len(prices) < 10:
            return {
                "microstructure_score": 0.5,
                "sub_scores":           {},
                "interpretation":       "Insufficient data",
                "data_quality":         "none",
            }
        # Simulate OHLC from daily closes (rough but functional)
        atr_pct = float(stock_data.get("atr_pct", 0.02))
        hl_range = prices * atr_pct
        d = {
            "open":   prices * 0.998,
            "high":   prices + hl_range * 0.6,
            "low":    prices - hl_range * 0.4,
            "close":  prices,
            "volume": volumes,
        }
        data_quality = "partial"

    # Compute all sub-scores
    sub_scores = {
        "dollar_vol_accel": _dollar_vol_acceleration(d),
        "price_efficiency": _price_efficiency_ratio(d),
        "spread_proxy":     _spread_proxy(d),
        "accum_dist":       _accumulation_distribution(d),
        "gap_analysis":     _gap_analysis(d),
        "price_impact":     _price_impact(d),
        "vwap_drift":       _vwap_drift(d),
        "wick_analysis":    _wick_analysis(d),
    }

    # Weighted composite
    score = float(sum(
        _WEIGHTS.get(k, 0.0) * v
        for k, v in sub_scores.items()
    ))
    score = float(np.clip(score, 0.0, 1.0))

    # Store for historical tracking
    history_store.update_stock(ticker, None, {"microstructure": score})

    # Human-readable interpretation
    if score >= 0.70:
        interpretation = (
            "Strong institutional accumulation signature — dollar volume accelerating, "
            "up-day volume dominating, price accepting higher levels."
        )
    elif score >= 0.55:
        interpretation = (
            "Positive tape — price efficiency and volume patterns suggest "
            "controlled buying rather than noise."
        )
    elif score >= 0.45:
        interpretation = "Neutral tape — no clear institutional footprint."
    elif score >= 0.30:
        interpretation = (
            "Cautious tape — some distribution signals present. "
            "Widen stops or reduce conviction."
        )
    else:
        interpretation = (
            "Weak microstructure — volume patterns suggest selling pressure "
            "or thinning liquidity. High-risk environment."
        )

    return {
        "microstructure_score": round(score, 3),
        "sub_scores":           {k: round(v, 3) for k, v in sub_scores.items()},
        "interpretation":       interpretation,
        "data_quality":         data_quality,
    }
