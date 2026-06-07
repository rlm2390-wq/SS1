# ─────────────────────────────────────────────
#  trade_setup.py  –  Trade Setup Card Engine
#
#  Generates a complete, actionable trade plan
#  from a scan result dict. No opinions —
#  pure math from ATR, price levels, and scores.
#
#  Rules:
#    Entry  = breakout above resistance + 0.3% buffer
#    Stop   = entry − 1.5 × ATR(14)
#    T1     = entry + 2 × (entry − stop)   [2R]
#    T2     = entry + 3 × (entry − stop)   [3R]
#    Size   = from position_size dict already computed
#    Invalidation = close below stop − 0.5% buffer
# ─────────────────────────────────────────────
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np


# ── Resistance detection ──────────────────────────────────────────────────────

def _find_resistance(prices: np.ndarray, current: float) -> float:
    """
    Find the nearest significant resistance level above current price.
    Uses a pivot-high approach over the last 60 days.
    Falls back to 52-week high if no clear pivot found.
    """
    if len(prices) < 10:
        return current * 1.03   # flat fallback

    window = min(len(prices), 60)
    recent = prices[-window:]
    n      = len(recent)

    # Find pivot highs: a bar where high > surrounding 3 bars on each side
    pivot_highs = []
    for i in range(3, n - 3):
        if recent[i] == max(recent[i-3:i+4]):
            pivot_highs.append(float(recent[i]))

    # Filter: only pivots above current price
    above = [p for p in pivot_highs if p > current * 1.001]

    if above:
        return min(above)   # nearest resistance

    # No pivot above — use recent high + small buffer
    recent_high = float(recent[-20:].max()) if len(recent) >= 20 else float(recent.max())
    if recent_high > current:
        return recent_high
    return current * 1.03


def _find_support(prices: np.ndarray, current: float) -> float:
    """Find nearest support below current price using pivot lows."""
    if len(prices) < 10:
        return current * 0.95

    window = min(len(prices), 60)
    recent = prices[-window:]
    n      = len(recent)

    pivot_lows = []
    for i in range(3, n - 3):
        if recent[i] == min(recent[i-3:i+4]):
            pivot_lows.append(float(recent[i]))

    below = [p for p in pivot_lows if p < current * 0.999]
    if below:
        return max(below)

    return current * 0.95


# ── Setup type to entry style map ─────────────────────────────────────────────

_SETUP_NOTES = {
    "volatility_breakout": (
        "Price coiling near highs — enter on volume breakout above resistance. "
        "Wait for a close above the entry zone, not just an intraday touch."
    ),
    "trend_pullback": (
        "Uptrend pulling back to support — enter near MA support on declining volume. "
        "A reversal candle (hammer, engulfing) adds conviction."
    ),
    "short_squeeze": (
        "Short squeeze developing — enter on price turning up with volume surge. "
        "Move can be fast and violent — honor your stop aggressively."
    ),
    "earnings_drift": (
        "Post-earnings drift — enter on continued institutional accumulation. "
        "Avoid chasing opening gaps — wait for intraday pullback."
    ),
}

_REGIME_SIZING = {
    "risk_on_strong": 1.00,
    "risk_on":        0.85,
    "choppy":         0.60,
    "risk_off":       0.40,
    "panic":          0.20,
}


# ── Public API ────────────────────────────────────────────────────────────────

def compute_trade_setup(result: Dict[str, Any], stock_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a complete trade setup card from a scan result.

    Returns a dict with:
      entry_zone, stop, target1, target2, risk_per_share,
      reward1, reward2, rr1, rr2, position_size, setup_notes,
      invalidation, regime_size_factor, atr14, confidence
    """
    prices   = stock_data.get("prices",  np.array([]))
    last     = float(result.get("last_price", 0))
    atr_pct  = float(stock_data.get("atr_pct",  0.02))
    atr14    = last * atr_pct
    regime   = result.get("regime", "choppy")
    setups   = result.get("setups", [])
    upside   = result.get("upside",  0.5)
    risk_sc  = result.get("risk",    0.5)

    if last <= 0:
        return {"error": "No price data"}

    # ── Entry zone ────────────────────────────────────────────────────────────
    resistance   = _find_resistance(prices, last)
    entry_low    = round(resistance * 1.002, 2)   # +0.2% above resistance
    entry_high   = round(resistance * 1.005, 2)   # +0.5% buffer
    entry_mid    = round((entry_low + entry_high) / 2, 2)

    # ── Stop ──────────────────────────────────────────────────────────────────
    stop         = round(entry_mid - 1.5 * atr14, 2)
    stop         = max(stop, last * 0.80)    # floor at 20% below last
    risk_per_sh  = round(entry_mid - stop, 2)

    if risk_per_sh <= 0:
        risk_per_sh = round(atr14 * 1.5, 2)
        stop        = round(entry_mid - risk_per_sh, 2)

    # ── Targets ───────────────────────────────────────────────────────────────
    t1           = round(entry_mid + 2.0 * risk_per_sh, 2)   # 2R
    t2           = round(entry_mid + 3.0 * risk_per_sh, 2)   # 3R

    # ── R/R ratios ────────────────────────────────────────────────────────────
    rr1          = round((t1 - entry_mid) / risk_per_sh, 1)
    rr2          = round((t2 - entry_mid) / risk_per_sh, 1)

    # ── Position size (regime-adjusted) ──────────────────────────────────────
    ps           = result.get("position_size", {})
    base_shares  = int(ps.get("suggested_shares", 0))
    regime_mult  = _REGIME_SIZING.get(regime, 0.60)
    adj_shares   = max(1, int(base_shares * regime_mult))
    dollar_risk  = round(adj_shares * risk_per_sh, 2)
    dollar_value = round(adj_shares * entry_mid, 2)

    # ── Invalidation ─────────────────────────────────────────────────────────
    invalidation = round(stop * 0.995, 2)   # 0.5% below stop

    # ── Setup notes ───────────────────────────────────────────────────────────
    primary_setup = setups[0] if setups else None
    notes = _SETUP_NOTES.get(primary_setup,
        "Signal-driven entry — wait for price to enter the entry zone "
        "with above-average volume before committing."
    )

    # ── Regime warning ────────────────────────────────────────────────────────
    regime_warning = None
    if regime in ("choppy", "risk_off", "panic"):
        regime_warning = {
            "choppy":   f"Choppy market — reduce size to {int(regime_mult*100)}% of normal. Favor defined-risk options plays.",
            "risk_off": f"Risk-off environment — size at {int(regime_mult*100)}%. Widen stops or wait for regime improvement.",
            "panic":    f"Panic regime — size at {int(regime_mult*100)}%. Only highest-conviction setups. Honor stops immediately.",
        }[regime]

    # ── Confidence (0–5) ─────────────────────────────────────────────────────
    confidence = 0
    if upside  >= 0.70: confidence += 2
    elif upside >= 0.60: confidence += 1
    if setups:           confidence += 1
    if result.get("pre_signals", {}).get("signal_count", 0) >= 2: confidence += 1
    if result.get("microstructure", {}).get("microstructure_score", 0) >= 0.65: confidence += 1
    confidence = min(confidence, 5)

    return {
        "ticker":          result.get("ticker"),
        "last_price":      round(last, 2),
        "atr14":           round(atr14, 2),
        "atr_pct":         round(atr_pct * 100, 2),
        "resistance":      round(resistance, 2),
        "entry_low":       entry_low,
        "entry_high":      entry_high,
        "entry_mid":       entry_mid,
        "stop":            stop,
        "invalidation":    invalidation,
        "target1":         t1,
        "target2":         t2,
        "risk_per_share":  risk_per_sh,
        "reward1":         round(t1 - entry_mid, 2),
        "reward2":         round(t2 - entry_mid, 2),
        "rr1":             rr1,
        "rr2":             rr2,
        "suggested_shares": adj_shares,
        "dollar_risk":     dollar_risk,
        "dollar_value":    dollar_value,
        "regime":          regime,
        "regime_size_pct": int(regime_mult * 100),
        "regime_warning":  regime_warning,
        "setup_notes":     notes,
        "primary_setup":   primary_setup,
        "confidence":      confidence,
        "entry_type":      "Breakout confirmation",
        "stop_type":       "1.5× ATR(14)",
        "target_type":     "2R / 3R",
    }
