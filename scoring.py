# ─────────────────────────────────────────────
#  scoring.py  –  Regime-Adaptive Weighting Engine
#
#  Replaces the simple correlation-based weights
#  with a full regime-aware adaptive system:
#
#  1. Detects regime CHARACTER (momentum / value /
#     defensive / panic) from factor behavior —
#     not just the trend label
#  2. Maintains per-regime weight matrices that
#     shift based on what's been predictive
#  3. Bounds weights so no single factor dominates
#  4. Tracks confidence — new regimes = fall back
#     toward equal weights until proven
#  5. Exposes current weights for header display
# ─────────────────────────────────────────────
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from config import (
    ADAPTIVE_WEIGHT_ENABLED, ADAPTIVE_WEIGHT_WINDOW,
    SCORING_WEIGHTS,
)
from history import HistoryStore

logger = logging.getLogger("scoring")

# ── Weight bounds — no factor goes below MIN or above MAX ─────────────────────
_WEIGHT_MIN = 0.05
_WEIGHT_MAX = 0.40

# ── Regime character profiles ─────────────────────────────────────────────────
# These are the PRIOR weights per regime character.
# They are blended with the data-driven correlation weights.
# Values must sum to 0.90 (regime gets fixed 0.10).
_REGIME_PRIORS: Dict[str, Dict[str, float]] = {
    # Strong trending up — technicals dominate, sentiment matters
    "momentum": {
        "technical":   0.32,
        "fundamental": 0.18,
        "sentiment":   0.20,
        "structural":  0.12,
        "setup":       0.08,
    },
    # Earnings-driven / rotation — fundamentals dominate
    "value_rotation": {
        "technical":   0.15,
        "fundamental": 0.38,
        "sentiment":   0.15,
        "structural":  0.14,
        "setup":       0.08,
    },
    # Choppy / range-bound — setups and structure matter most
    "choppy": {
        "technical":   0.20,
        "fundamental": 0.20,
        "sentiment":   0.15,
        "structural":  0.20,
        "setup":       0.15,
    },
    # Risk-off / defensive — risk and structural dominate, deweight tech
    "defensive": {
        "technical":   0.12,
        "fundamental": 0.28,
        "sentiment":   0.12,
        "structural":  0.28,
        "setup":       0.10,
    },
    # Panic — everything compressed, risk filtering is paramount
    "panic": {
        "technical":   0.10,
        "fundamental": 0.20,
        "sentiment":   0.10,
        "structural":  0.35,
        "setup":       0.15,
    },
}

_FACTOR_KEYS = ["technical", "fundamental", "sentiment", "structural", "setup"]

# Module-level cache — persists across calls within a process
_cached_weights: Optional[Dict[str, float]] = None
_cached_regime_char: str = "momentum"
_cached_confidence: float = 0.0


def detect_regime_character(
    regime_label: str,
    history_store: HistoryStore,
) -> Tuple[str, float]:
    """
    Detect the CURRENT regime character from factor behavior history.

    Returns (character_label, confidence_0_to_1).

    Character is distinct from regime label:
    - "risk_on_strong" label could be "momentum" OR "value_rotation"
      depending on WHICH factors have been driving scores up.
    - Confidence reflects how stable the regime has been lately.
    """
    # Map label → default character
    label_defaults = {
        "risk_on_strong": "momentum",
        "risk_on":        "momentum",
        "choppy":         "choppy",
        "risk_off":       "defensive",
        "panic":          "panic",
    }
    default_char = label_defaults.get(regime_label, "choppy")

    # Get 20-day factor score history for all tickers
    tech_hist   = history_store.get_all_stock_values("technical",   lookback_days=20)
    fund_hist   = history_store.get_all_stock_values("fundamental", lookback_days=20)

    if len(tech_hist) < 10 or len(fund_hist) < 10:
        return default_char, 0.3   # not enough data yet

    tech_mean = float(np.mean(tech_hist))
    fund_mean = float(np.mean(fund_hist))

    # Determine if technicals or fundamentals are driving scores higher
    tech_vs_fund = tech_mean - fund_mean

    if regime_label in ("risk_on_strong", "risk_on"):
        if tech_vs_fund > 0.08:
            char = "momentum"
        elif tech_vs_fund < -0.08:
            char = "value_rotation"
        else:
            char = "momentum"
    else:
        char = default_char

    # Confidence: how long has this regime label been stable?
    regime_history = history_store.get_market_history("regime_label_hash", lookback_days=30)
    if len(regime_history) < 3:
        confidence = 0.30
    else:
        # Count fraction of recent scans with same regime label
        label_hash = hash(regime_label) % 1000
        stability  = float(np.mean([abs(v - label_hash) < 1 for v in regime_history]))
        confidence = float(np.clip(stability, 0.20, 0.95))

    return char, confidence


def _compute_adaptive_weights(
    regime_label: str,
    regime_char: str,
    confidence: float,
    history_store: HistoryStore,
) -> Dict[str, float]:
    """
    Blend three weight sources:
      1. Prior for the detected regime character
      2. Data-driven rolling factor-return correlations
      3. Static fallback weights from config

    Blend ratio is determined by confidence:
      - High confidence (0.8+): 60% prior + 40% data-driven
      - Low confidence (<0.3):  20% prior + 80% static fallback
    """
    prior_weights = _REGIME_PRIORS.get(regime_char, _REGIME_PRIORS["choppy"])

    # Compute data-driven correlation weights
    corr_weights: Dict[str, float] = {}
    for key in _FACTOR_KEYS:
        corr = history_store.cross_sectional_rolling_corr(
            factor_key=key,
            return_key="upside",
            window=ADAPTIVE_WEIGHT_WINDOW,
        )
        corr_weights[key] = max(0.0, corr)

    corr_total = sum(corr_weights.values())
    if corr_total < 0.05:
        # No meaningful correlation data — use prior + static blend
        for k in _FACTOR_KEYS:
            corr_weights[k] = SCORING_WEIGHTS.get(k, 0.18)
        corr_total = sum(corr_weights.values())

    # Normalize correlation weights to 0.90
    for k in _FACTOR_KEYS:
        corr_weights[k] = 0.90 * corr_weights[k] / (corr_total + 1e-9)

    # Blend: confidence drives how much we trust the regime prior
    prior_blend = float(np.clip(confidence * 0.60, 0.10, 0.60))
    corr_blend  = 1.0 - prior_blend

    blended: Dict[str, float] = {}
    for k in _FACTOR_KEYS:
        blended[k] = prior_blend * prior_weights[k] + corr_blend * corr_weights[k]

    # Enforce bounds
    for k in _FACTOR_KEYS:
        blended[k] = float(np.clip(blended[k], _WEIGHT_MIN, _WEIGHT_MAX))

    # Re-normalize to exactly 0.90
    total = sum(blended.values())
    for k in _FACTOR_KEYS:
        blended[k] = 0.90 * blended[k] / (total + 1e-9)

    blended["regime"] = 0.10
    return blended


def get_current_weights(
    regime_label: str,
    history_store: HistoryStore,
) -> Tuple[Dict[str, float], str, float]:
    """
    Public API: return (weights, regime_char, confidence).
    Caches the result so repeated calls within a scan are free.
    """
    global _cached_weights, _cached_regime_char, _cached_confidence

    if not ADAPTIVE_WEIGHT_ENABLED:
        static = dict(SCORING_WEIGHTS)
        static["regime"] = 0.10
        return static, "static", 1.0

    char, confidence = detect_regime_character(regime_label, history_store)

    weights = _compute_adaptive_weights(
        regime_label, char, confidence, history_store)

    _cached_weights      = weights
    _cached_regime_char  = char
    _cached_confidence   = confidence

    # Store regime label hash for stability tracking
    history_store.update_market(None, {
        "regime_label_hash": float(hash(regime_label) % 1000),
        "regime_confidence": confidence,
        "weight_tech":       weights.get("technical",   0),
        "weight_fund":       weights.get("fundamental", 0),
        "weight_sent":       weights.get("sentiment",   0),
        "weight_struct":     weights.get("structural",  0),
        "weight_setup":      weights.get("setup",       0),
    })

    logger.debug(
        "Regime: %s (%s) confidence=%.2f | tech=%.2f fund=%.2f sent=%.2f struct=%.2f setup=%.2f",
        regime_label, char, confidence,
        weights["technical"], weights["fundamental"],
        weights["sentiment"], weights["structural"], weights["setup"],
    )

    return weights, char, confidence


def get_top2_weights(weights: Dict[str, float]) -> str:
    """
    Return a display string for the header showing the top 2 weighted factors.
    e.g. "Tech 34% · Fund 28%"
    """
    display_names = {
        "technical":   "Tech",
        "fundamental": "Fund",
        "sentiment":   "Sent",
        "structural":  "Struct",
        "setup":       "Setup",
        "regime":      "Regime",
    }
    sorted_w = sorted(
        [(k, v) for k, v in weights.items() if k != "regime"],
        key=lambda x: x[1], reverse=True
    )
    top2 = sorted_w[:2]
    return " · ".join(
        f"{display_names.get(k, k)} {round(v * 100)}%"
        for k, v in top2
    )


def compute_upside_score(
    tech_score: float,
    fund_score: float,
    sent_score: float,
    struct_score: float,
    setup_score: float,
    regime_score: float,
    history_store: HistoryStore,
    regime_label: str = "unknown",
    microstructure_score: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute UpsideScore using regime-adaptive weights.

    Returns (upside_score, weights_used) so callers can display
    the active weight distribution.

    microstructure_score (0–1) from the Microstructure Brain
    adds a small overlay bonus/penalty without changing the weights.
    """
    weights, char, confidence = get_current_weights(regime_label, history_store)

    scores = {
        "technical":   tech_score,
        "fundamental": fund_score,
        "sentiment":   sent_score,
        "structural":  struct_score,
        "setup":       setup_score,
        "regime":      regime_score,
    }

    upside = float(sum(weights.get(k, 0.0) * v for k, v in scores.items()))

    # Microstructure overlay: ±3% based on tape quality
    micro_adj = (microstructure_score - 0.5) * 0.06   # range: -3% to +3%
    upside    = float(np.clip(upside + micro_adj, 0.0, 1.0))

    # Regime gate
    if regime_score < 0.30:
        upside *= 0.70
    elif regime_score > 0.75:
        upside = float(np.clip(upside * 1.05, 0.0, 1.0))

    return float(np.clip(upside, 0.0, 1.0)), weights
