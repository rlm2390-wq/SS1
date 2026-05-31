# ─────────────────────────────────────────────
#  brains/regime.py  –  Market regime brain
# ─────────────────────────────────────────────
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any

from storage.history import HistoryStore


# ── Sub-scores ────────────────────────────────────────────────────────────────

def _index_trend_score(index_data: Dict[str, Any]) -> float:
    """Score the index trend: price vs MAs + MA slopes.  Returns 0–1."""
    spy   = index_data["spy_prices"]
    ma50  = index_data["spy_ma50"]
    ma200 = index_data["spy_ma200"]

    last = spy[-1]
    score = 0.0

    # Price vs MAs
    if not np.isnan(ma50[-1])  and last > ma50[-1]:  score += 0.25
    if not np.isnan(ma200[-1]) and last > ma200[-1]: score += 0.25

    # MA slopes (20-day change)
    if not np.isnan(ma50[-1])  and not np.isnan(ma50[-21])  and ma50[-1]  > ma50[-21]:  score += 0.25
    if not np.isnan(ma200[-1]) and not np.isnan(ma200[-21]) and ma200[-1] > ma200[-21]: score += 0.25

    return float(np.clip(score, 0.0, 1.0))


def _breadth_score(index_data: Dict[str, Any]) -> float:
    """Score market breadth (% of stocks above 50 MA).  Returns 0–1."""
    breadth = index_data.get("breadth_pct_above_50ma", 0.5)
    # 0–1 fraction → sigmoid-style mapping
    # 0.70+ = great, 0.50 = neutral, 0.30- = weak
    raw = (breadth - 0.30) / 0.40   # maps [0.30, 0.70] → [0, 1]
    return float(np.clip(raw, 0.0, 1.0))


def _vix_score(index_data: Dict[str, Any]) -> float:
    """
    Score volatility regime.
    Lower VIX relative to its own recent average → higher score (calmer = better for longs).
    Returns 0–1.
    """
    vix_cur = index_data.get("vix_current", 20.0)
    vix_avg = index_data.get("vix_20d_avg", 20.0)

    ratio = vix_cur / max(vix_avg, 1e-6)
    # ratio: 0.5 (very calm) → 1.0, 2.0 (panic) → 0.0
    score = (2.0 - ratio) / 1.5
    return float(np.clip(score, 0.0, 1.0))


def _put_call_score(index_data: Dict[str, Any]) -> float:
    """
    Score investor sentiment via put/call ratio.
    Low P/C ratio → bullish sentiment.  Returns 0–1.
    """
    pc = index_data.get("put_call_ratio", 0.85)
    # 0.5 (very bullish) → 1.0,  1.5+ (very bearish) → 0.0
    score = (1.5 - pc) / 1.0
    return float(np.clip(score, 0.0, 1.0))


# ── Regime label logic ────────────────────────────────────────────────────────

def _classify_regime(trend: float, breadth: float, vix: float, pc: float) -> str:
    composite = 0.35 * trend + 0.25 * breadth + 0.25 * vix + 0.15 * pc

    if composite >= 0.72:
        return "risk_on_strong"
    elif composite >= 0.55:
        return "risk_on"
    elif composite >= 0.40:
        return "choppy"
    elif composite >= 0.25:
        return "risk_off"
    else:
        return "panic"


# ── Public API ────────────────────────────────────────────────────────────────

def compute_market_context(
    index_data: Dict[str, Any],
    history_store: HistoryStore,
) -> Tuple[str, float]:
    """
    Compute market regime label and 0–1 regime score.

    Steps
    -----
    1. Score four sub-components: trend, breadth, vix, put/call.
    2. Blend into composite regime_score.
    3. Apply z-score vs recent history to detect regime shifts.
    4. Classify into a human-readable label.

    Returns
    -------
    (regime_label, regime_score)
    """
    trend   = _index_trend_score(index_data)
    breadth = _breadth_score(index_data)
    vix     = _vix_score(index_data)
    pc      = _put_call_score(index_data)

    # Weighted blend
    regime_score = float(
        0.35 * trend +
        0.25 * breadth +
        0.25 * vix +
        0.15 * pc
    )

    # Adjust score via z-score vs historical regime scores
    history = history_store.get_market_history("regime_score", lookback_days=90)
    if len(history) >= 10:
        z = history_store.zscore(regime_score, history)
        # Nudge score slightly: if historically high, bump up a touch; if low, down
        regime_score = float(np.clip(regime_score + 0.05 * z, 0.0, 1.0))

    label = _classify_regime(trend, breadth, vix, pc)

    # Persist for future z-scoring
    history_store.update_market(None, {
        "regime_score": regime_score,
        "trend_score":  trend,
        "breadth_score": breadth,
        "vix_score":    vix,
    })

    return label, regime_score
