# ─────────────────────────────────────────────
#  brains/technical.py  –  Technical brain
# ─────────────────────────────────────────────
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple

from history import HistoryStore


# ── Raw factor computation ────────────────────────────────────────────────────

def compute_technical_factors(
    stock_data: Dict[str, Any],
    history_store: HistoryStore,
) -> Dict[str, float]:
    """
    Compute raw (un-normalized) technical metrics.

    Returns a dict of named metrics that score_technical_factors() will normalize.
    """
    prices  = stock_data["prices"]
    volumes = stock_data["volumes"]
    ma20    = stock_data["ma20"]
    ma50    = stock_data["ma50"]
    ma200   = stock_data["ma200"]
    rsi     = stock_data["rsi"]
    bb_w    = stock_data["bb_width"]

    last = float(prices[-1])
    n    = len(prices)

    # ── Trend ────────────────────────────────────────────────────────────────
    # Fraction of [price, ma50, ma200] that are in bullish alignment
    ma20_last  = float(ma20[-1])  if not np.isnan(ma20[-1])  else last
    ma50_last  = float(ma50[-1])  if not np.isnan(ma50[-1])  else last
    ma200_last = float(ma200[-1]) if not np.isnan(ma200[-1]) else last

    above_ma20  = float(last > ma20_last)
    above_ma50  = float(last > ma50_last)
    above_ma200 = float(last > ma200_last)

    # MA slope: 20-day change normalized by price
    def _slope(ma, lookback=20):
        valid = ma[~np.isnan(ma)]
        if len(valid) < lookback + 1:
            return 0.0
        return float((valid[-1] - valid[-lookback - 1]) / (valid[-lookback - 1] + 1e-9))

    ma50_slope  = _slope(ma50,  20)
    ma200_slope = _slope(ma200, 20)

    # Composite trend metric: positive = bullish
    trend_raw = (above_ma20 + above_ma50 + above_ma200) / 3.0 + 0.5 * max(ma50_slope, 0)

    # ── Volatility / compression ──────────────────────────────────────────────
    # Compare recent 20-day vol to 60-day vol
    if n >= 60:
        vol_20 = float(np.std(np.diff(prices[-21:]) / prices[-21:-1]))
        vol_60 = float(np.std(np.diff(prices[-61:]) / prices[-61:-1]))
        vol_ratio = vol_20 / max(vol_60, 1e-9)
    else:
        vol_ratio = 1.0

    # Also capture Bollinger Band width percentile signal
    bb_width_raw = float(bb_w)   # narrower = more compressed = setup potential

    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi_raw = float(rsi)

    # ── Volume / accumulation ─────────────────────────────────────────────────
    avg_vol_20 = float(volumes[-20:].mean())
    avg_vol_60 = float(volumes[-60:].mean()) if n >= 60 else avg_vol_20
    vol_surge  = avg_vol_20 / max(avg_vol_60, 1e-9)

    # Up-volume vs down-volume balance (last 20 bars)
    price_changes = np.diff(prices[-21:])
    vols_20 = volumes[-20:]
    up_vol   = float(vols_20[price_changes > 0].sum())
    down_vol = float(vols_20[price_changes < 0].sum())
    accumulation = up_vol / max(up_vol + down_vol, 1e-9)  # 1.0 = all up-volume

    # ── 52-week proximity ─────────────────────────────────────────────────────
    high_52w = float(prices[-252:].max()) if n >= 252 else float(prices.max())
    low_52w  = float(prices[-252:].min()) if n >= 252 else float(prices.min())
    rng_52w  = high_52w - low_52w
    pos_52w  = (last - low_52w) / max(rng_52w, 1e-9)   # 0=at 52w low, 1=at high

    return {
        "trend_raw":       trend_raw,
        "vol_ratio":       vol_ratio,
        "bb_width":        bb_width_raw,
        "rsi":             rsi_raw,
        "vol_surge":       vol_surge,
        "accumulation":    accumulation,
        "pos_52w":         pos_52w,
        "ma50_slope":      ma50_slope,
    }


# ── Score normalization ───────────────────────────────────────────────────────

def score_technical_factors(
    raw_factors: Dict[str, float],
    history_store: HistoryStore,
) -> Tuple[float, Dict[str, float]]:
    """
    Normalize raw technical metrics into 0–1 sub-scores, then blend.

    Normalization uses cross-sectional percentile ranks from history_store
    where available, otherwise falls back to hand-calibrated thresholds.
    """

    def pct_rank(key: float, hist_key: str, default_50: float) -> float:
        history = history_store.get_all_stock_values(hist_key, lookback_days=60)
        if len(history) >= 10:
            return history_store.percentile_rank(key, history)
        # Fallback: sigmoid around a neutral value
        return float(np.clip(0.5 + (key - default_50) / (2 * abs(default_50) + 1e-9), 0.0, 1.0))

    # ── Trend score ───────────────────────────────────────────────────────────
    # trend_raw in roughly [0, 1.5] where 1.0+ = strongly bullish
    trend_score = float(np.clip(raw_factors["trend_raw"] / 1.2, 0.0, 1.0))

    # ── Volatility / compression score ───────────────────────────────────────
    # Low vol_ratio = compression → high setup potential
    vr = raw_factors["vol_ratio"]
    if   vr < 0.60: vol_score = 1.0
    elif vr < 0.80: vol_score = 0.80
    elif vr < 1.00: vol_score = 0.60
    elif vr < 1.30: vol_score = 0.35
    else:           vol_score = 0.10
    vol_score = float(vol_score)

    # ── RSI score ─────────────────────────────────────────────────────────────
    # Oversold in an uptrend is ideal; overbought penalized
    rsi = raw_factors["rsi"]
    ts  = trend_score
    if   ts >= 0.65 and 30 <= rsi <= 50: rsi_score = 1.00
    elif ts >= 0.65 and 50 < rsi <= 65:  rsi_score = 0.75
    elif ts >= 0.65 and 65 < rsi <= 75:  rsi_score = 0.50
    elif rsi < 30:                        rsi_score = 0.45   # oversold but could be broken
    elif rsi > 80:                        rsi_score = 0.15
    else:                                 rsi_score = 0.50
    rsi_score = float(rsi_score)

    # ── Volume score ──────────────────────────────────────────────────────────
    surge = raw_factors["vol_surge"]
    accum = raw_factors["accumulation"]
    vol_acc_score = float(np.clip(0.5 * (surge - 1.0) / 1.0 + accum, 0.0, 1.0))

    # ── 52-week position ──────────────────────────────────────────────────────
    pos = raw_factors["pos_52w"]
    # Near 52w high (0.85+) = potential breakout OR extended; prefer 0.50–0.90
    if   pos >= 0.85: pos_score = 0.75
    elif pos >= 0.60: pos_score = 0.90
    elif pos >= 0.40: pos_score = 0.65
    else:             pos_score = 0.30
    pos_score = float(pos_score)

    # ── Composite tech score ──────────────────────────────────────────────────
    sub_scores = {
        "trend":      trend_score,
        "volatility": vol_score,
        "rsi":        rsi_score,
        "volume_acc": vol_acc_score,
        "pos_52w":    pos_score,
    }

    tech_score = float(
        0.30 * trend_score +
        0.20 * vol_score +
        0.20 * rsi_score +
        0.20 * vol_acc_score +
        0.10 * pos_score
    )

    return float(np.clip(tech_score, 0.0, 1.0)), sub_scores
