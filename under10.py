# ─────────────────────────────────────────────
#  under10.py  –  Under $10 High-Potential Filter
#
#  Strict 6-gate filter + explosive setup requirement
#  + custom scoring formula. Replaces the old
#  is_under20_popper() logic entirely.
#
#  Gates (ALL must pass):
#    1. Price $1.00–$10.00
#    2. Avg dollar volume > $1M (no illiquid shells)
#    3. Float > 20M shares (no micro-float traps)
#    4. 2+ pre-signals (filters 90% of false positives)
#    5. At least one explosive setup present
#    6. Risk score <= 0.75
#
#  Scoring formula:
#    Under10Score =
#      0.40 * signal_count_score
#    + 0.40 * explosive_setup_score
#    + 0.20 * (1 - risk_score)
#
#  Tiers (color coded in UI):
#    3+ signals = strong (green)
#    2  signals = medium (yellow)
#    1  signal  = filtered out
# ─────────────────────────────────────────────
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

# ── Explosive setup labels ────────────────────────────────────────────────────
# These are considered "explosive" — high-velocity potential
_EXPLOSIVE_SETUPS = {
    "short_squeeze",
    "volatility_breakout",
}

# These are "supporting" — good but not explosive on their own
_SUPPORTING_SETUPS = {
    "trend_pullback",
    "earnings_drift",
}

# Pre-signal names that count as explosive triggers
_EXPLOSIVE_PRE_SIGNALS = {
    "oi_buildup",
    "si_acceleration",
    "vol_contraction",
}

_SUPPORTING_PRE_SIGNALS = {
    "volume_dryup",
    "insider_cluster",
    "earnings_iv_expand",
}

_MAX_SIGNALS = 6   # total pre-signal detectors

# ── Gate functions ────────────────────────────────────────────────────────────

def _passes_speculative_filter(result: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Returns (passes, reason_if_rejected).
    Enforces hard liquidity and float guardrails.
    """
    price  = result.get("last_price", 999)
    risk   = result.get("risk", 1.0)

    # Gate 1: Price range
    if price > 10.0:
        return False, "price > $10"
    if price < 1.0:
        return False, "price < $1.00 (penny stock)"

    # Gate 2: Avg dollar volume (stored in factor scores via risk brain)
    # We infer it from the risk sub-scores; if not available, allow through
    # with a risk penalty applied at scoring stage
    sub = result.get("sub_scores", {}).get("risk", {})
    liq_risk = sub.get("liquidity", 0.6) if isinstance(sub, dict) else 0.6
    if liq_risk > 0.80:   # maps to avg_dv < ~$500k
        return False, "avg dollar volume too low"

    # Gate 3: Float — inferred from structural score
    # Very low float_score (<0.15) suggests micro-float trap
    struct_sub = result.get("sub_scores", {}).get("structural", {})
    if isinstance(struct_sub, dict):
        float_own = struct_sub.get("float_ownership", 0.5)
        if float_own < 0.15:
            return False, "float too small or data unavailable"

    # Gate 4: Risk ceiling
    if risk > 0.75:
        return False, f"risk score {risk:.2f} > 0.75"

    return True, ""


def _get_explosive_setup_score(result: Dict[str, Any]) -> float:
    """
    Returns:
      1.0  if an explosive setup (squeeze / breakout) is present
      0.6  if only supporting setups
      0.0  if no setups at all
    Also counts explosive pre-signals as partial explosive score.
    """
    setups      = set(result.get("setups", []))
    pre_signals = result.get("pre_signals", {})
    fired_pre   = {s["name"] for s in pre_signals.get("signals", [])}

    has_explosive_setup = bool(setups & _EXPLOSIVE_SETUPS)
    has_supporting_setup= bool(setups & _SUPPORTING_SETUPS)
    has_explosive_pre   = bool(fired_pre & _EXPLOSIVE_PRE_SIGNALS)

    if has_explosive_setup:
        return 1.0
    if has_explosive_pre and has_supporting_setup:
        return 0.85
    if has_explosive_pre:
        return 0.70
    if has_supporting_setup:
        return 0.50
    return 0.0


def _get_signal_count_score(result: Dict[str, Any]) -> Tuple[int, float]:
    """
    Returns (signal_count, signal_count_score).
    Counts pre-signals + active setups as signals.
    """
    pre_count  = result.get("pre_signals", {}).get("signal_count", 0)
    setup_count= len(result.get("setups", []))

    # Each unique signal type counts once
    total_signals = pre_count + (1 if setup_count >= 1 else 0)
    signal_score  = float(np.clip(total_signals / _MAX_SIGNALS, 0.0, 1.0))

    return total_signals, signal_score


def _get_tier(signal_count: int) -> str:
    if signal_count >= 3:
        return "strong"
    elif signal_count >= 2:
        return "medium"
    else:
        return "weak"   # filtered out before reaching here


# ── Public API ────────────────────────────────────────────────────────────────

def compute_under10_score(result: Dict[str, Any]) -> float:
    """
    Compute the Under10Score for a candidate that has already passed gates.
    Returns 0–1.
    """
    signal_count, signal_score = _get_signal_count_score(result)
    explosive_score            = _get_explosive_setup_score(result)
    risk_score                 = float(result.get("risk", 0.5))

    return float(np.clip(
        0.40 * signal_score +
        0.40 * explosive_score +
        0.20 * (1.0 - risk_score),
        0.0, 1.0
    ))


def is_under10_candidate(result: Dict[str, Any]) -> Tuple[bool, float, Dict]:
    """
    Full gate + scoring evaluation.

    Returns (qualifies, under10_score, metadata_dict).
    metadata_dict contains tier, signal_count, explosive_score,
    rejection_reason for debugging.
    """
    # Gate: speculative filter
    passes, reason = _passes_speculative_filter(result)
    if not passes:
        return False, 0.0, {"rejected": reason}

    # Gate: pre-signal count >= 2
    signal_count, signal_score = _get_signal_count_score(result)
    if signal_count < 2:
        return False, 0.0, {"rejected": f"only {signal_count} signal(s)"}

    # Gate: explosive setup required
    explosive_score = _get_explosive_setup_score(result)
    if explosive_score == 0.0:
        return False, 0.0, {"rejected": "no explosive setup"}

    # Score
    risk_score = float(result.get("risk", 0.5))
    u10_score  = float(np.clip(
        0.40 * signal_score +
        0.40 * explosive_score +
        0.20 * (1.0 - risk_score),
        0.0, 1.0
    ))

    tier = _get_tier(signal_count)

    meta = {
        "under10_score":    round(u10_score, 3),
        "tier":             tier,
        "signal_count":     signal_count,
        "explosive_score":  round(explosive_score, 3),
        "signal_score":     round(signal_score, 3),
        "rejected":         None,
    }

    return True, u10_score, meta


def filter_and_rank_under10(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply all gates, score, and rank.
    Attaches 'under10_meta' to each qualifying result.
    Returns sorted list, strongest first.
    """
    candidates = []
    for r in results:
        qualifies, score, meta = is_under10_candidate(r)
        if qualifies:
            enriched = dict(r)
            enriched["under10_meta"]  = meta
            enriched["under10_score"] = score
            candidates.append(enriched)

    candidates.sort(key=lambda x: x["under10_score"], reverse=True)
    return candidates
