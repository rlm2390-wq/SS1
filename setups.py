# ─────────────────────────────────────────────
#  brains/setups.py  –  Setup detection brain
#
#  Detects pre-move patterns using statistical
#  thresholds rather than hard-coded rules.
# ─────────────────────────────────────────────
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Tuple

from storage.history import HistoryStore


# ── Individual setup detectors ────────────────────────────────────────────────

def _detect_volatility_breakout(
    stock_data: Dict[str, Any],
    factor_scores: Dict[str, float],
) -> Tuple[bool, float]:
    """
    Volatility breakout: tight price compression near 52-week high,
    volume drying up, technical score strong.
    Returns (detected, strength 0-1).
    """
    prices = stock_data["prices"]
    bb_w   = stock_data.get("bb_width", 0.05)
    ma20   = stock_data["ma20"]

    tech_score = factor_scores.get("technical", 0.5)

    # 52-week proximity
    high_52w  = float(prices[-252:].max()) if len(prices) >= 252 else float(prices.max())
    last      = float(prices[-1])
    near_high = (last / high_52w) >= 0.92   # within 8% of 52w high

    # Bollinger Band width: low = compressed
    # Compare current width to 90-day history (z-score approach)
    n = min(len(prices), 90)
    bb_hist = [
        float(np.std(prices[max(0, i-19):i+1]) * 4 / (prices[max(0, i-19):i+1].mean() + 1e-9))
        for i in range(n - 20, n)
        if i >= 19
    ]
    bb_pct = float(np.mean(np.array(bb_hist) >= bb_w)) if bb_hist else 0.5
    compressed = bb_pct >= 0.70   # current width is in bottom 30% of recent history

    # Volume drying up on the contraction
    vol_recent  = float(stock_data["volumes"][-5:].mean())
    vol_20d     = float(stock_data["volumes"][-20:].mean())
    vol_drying  = vol_recent < vol_20d * 0.80

    detected = bool(compressed and near_high and tech_score >= 0.55)
    strength = float(np.clip(
        0.30 * float(compressed) +
        0.30 * float(near_high) +
        0.20 * float(vol_drying) +
        0.20 * tech_score,
        0.0, 1.0
    ))
    return detected, strength


def _detect_trend_pullback(
    stock_data: Dict[str, Any],
    factor_scores: Dict[str, float],
) -> Tuple[bool, float]:
    """
    Healthy pullback in a strong uptrend:
    - Strong uptrend (price above rising MAs)
    - Price pulled back to 20/50 MA zone
    - RSI cooling off
    - Volume declining on pullback (healthy)
    """
    prices = stock_data["prices"]
    ma20   = stock_data["ma20"]
    ma50   = stock_data["ma50"]
    rsi    = stock_data["rsi"]

    tech_score = factor_scores.get("technical", 0.5)
    last       = float(prices[-1])
    ma20_last  = float(ma20[-1]) if not np.isnan(ma20[-1]) else last
    ma50_last  = float(ma50[-1]) if not np.isnan(ma50[-1]) else last

    # In uptrend: price above MA50
    in_uptrend = bool(last > ma50_last)

    # Near MA20/MA50 support (within 5%)
    near_ma20 = abs(last - ma20_last) / (ma20_last + 1e-9) < 0.05
    near_ma50 = abs(last - ma50_last) / (ma50_last + 1e-9) < 0.07
    near_support = near_ma20 or near_ma50

    # RSI cooling off (not overbought)
    rsi_ok = 25 <= rsi <= 55

    # Volume declining on pullback (last 5 days vs previous 10)
    vol5  = float(stock_data["volumes"][-5:].mean())
    vol10 = float(stock_data["volumes"][-15:-5].mean()) if len(stock_data["volumes"]) >= 15 else vol5
    healthy_pullback_vol = vol5 < vol10 * 0.90

    detected = bool(in_uptrend and near_support and rsi_ok and tech_score >= 0.50)
    strength = float(np.clip(
        0.30 * float(in_uptrend) +
        0.25 * float(near_support) +
        0.20 * float(rsi_ok) +
        0.15 * float(healthy_pullback_vol) +
        0.10 * tech_score,
        0.0, 1.0
    ))
    return detected, strength


def _detect_short_squeeze(
    stock_data: Dict[str, Any],
    factor_scores: Dict[str, float],
) -> Tuple[bool, float]:
    """
    Short squeeze setup:
    - High short interest + high days-to-cover
    - Price turning/trending up
    - Volume accelerating
    - Structurally elevated
    """
    si           = stock_data.get("short_interest", {})
    short_float  = float(si.get("short_float_pct", 5.0))
    days_cover   = float(si.get("days_to_cover", 2.0))
    borrow_cost  = float(si.get("borrow_cost_pct", 1.0))
    short_trend  = float(si.get("short_trend", 0.0))   # <0 = covering

    struct_score = factor_scores.get("structural", 0.5)
    tech_score   = factor_scores.get("technical", 0.5)

    prices  = stock_data["prices"]
    volumes = stock_data["volumes"]

    # Price trending up over last 10 days
    price_up = bool(prices[-1] > prices[-10]) if len(prices) >= 10 else False

    # Volume accelerating: last 5 days vs 20-day avg
    vol5   = float(volumes[-5:].mean())
    vol20  = float(volumes[-20:].mean())
    vol_up = vol5 > vol20 * 1.20   # 20% above average

    # Shorts potentially trapped
    high_short = short_float > 15.0
    hard_borrow = borrow_cost > 5.0
    covering    = short_trend < -0.05   # short interest falling = covering

    detected = bool(high_short and (days_cover > 3) and price_up and struct_score >= 0.55)
    strength = float(np.clip(
        0.25 * float(np.clip(short_float / 30.0, 0.0, 1.0)) +
        0.20 * float(np.clip(days_cover / 10.0, 0.0, 1.0)) +
        0.15 * float(hard_borrow) +
        0.15 * float(covering) +
        0.15 * float(vol_up) +
        0.10 * struct_score,
        0.0, 1.0
    ))
    return detected, strength


def _detect_earnings_drift(
    stock_data: Dict[str, Any],
    factor_scores: Dict[str, float],
) -> Tuple[bool, float]:
    """
    Post-earnings drift (PEAD): stock beat estimates + price hasn't fully moved yet.
    """
    fund = stock_data.get("fundamentals", {})
    surp = float(fund.get("earnings_surprise", 0.0))
    tech_score = factor_scores.get("technical", 0.5)
    fund_score = factor_scores.get("fundamental", 0.5)

    strong_beat = surp > 0.05   # >5% EPS surprise

    # Check that momentum is not already exhausted
    rsi = stock_data.get("rsi", 50.0)
    not_overbought = rsi < 70

    detected = bool(strong_beat and not_overbought and fund_score >= 0.55)
    strength = float(np.clip(
        0.40 * float(np.clip(surp / 0.15, 0.0, 1.0)) +
        0.30 * fund_score +
        0.30 * float(not_overbought),
        0.0, 1.0
    ))
    return detected, strength


# ── Public API ────────────────────────────────────────────────────────────────

def detect_setups(
    stock_data: Dict[str, Any],
    factor_scores: Dict[str, float],
    regime_label: str,
    history_store: HistoryStore,
) -> Tuple[List[str], float]:
    """
    Run all setup detectors and return active setup labels + aggregate score.

    Each setup is scored 0–1; the composite setup_score is a weighted max
    (not average) so one very strong setup fires even if others are absent.
    """
    detected_setups: List[str] = []
    strengths: Dict[str, float] = {}

    # Adjust detection sensitivity by regime
    regime_penalty = 1.0
    if regime_label in ("panic", "risk_off"):
        regime_penalty = 0.70
    elif regime_label in ("choppy",):
        regime_penalty = 0.85

    vb_det, vb_str = _detect_volatility_breakout(stock_data, factor_scores)
    tp_det, tp_str = _detect_trend_pullback(stock_data, factor_scores)
    sq_det, sq_str = _detect_short_squeeze(stock_data, factor_scores)
    ed_det, ed_str = _detect_earnings_drift(stock_data, factor_scores)

    if vb_det:
        detected_setups.append("volatility_breakout")
        strengths["volatility_breakout"] = vb_str * regime_penalty
    if tp_det:
        detected_setups.append("trend_pullback")
        strengths["trend_pullback"] = tp_str * regime_penalty
    if sq_det:
        detected_setups.append("short_squeeze")
        strengths["short_squeeze"] = sq_str * regime_penalty
    if ed_det:
        detected_setups.append("earnings_drift")
        strengths["earnings_drift"] = ed_str * regime_penalty

    if not strengths:
        return [], 0.0

    # Composite: weighted toward the strongest setup, but reward multiple setups
    vals = sorted(strengths.values(), reverse=True)
    setup_score = vals[0]  # lead with the best
    for i, v in enumerate(vals[1:], 1):
        setup_score += v * (0.25 ** i)   # diminishing credit for additional setups
    setup_score = float(np.clip(setup_score, 0.0, 1.0))

    return detected_setups, setup_score
