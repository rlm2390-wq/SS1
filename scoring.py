# ─────────────────────────────────────────────
#  brains/scoring.py  –  UpsideScore engine
# ─────────────────────────────────────────────
from __future__ import annotations
import numpy as np
from typing import Dict
from config import SCORING_WEIGHTS, ADAPTIVE_WEIGHT_ENABLED, ADAPTIVE_WEIGHT_WINDOW
from history import HistoryStore


def _adaptive_weights(history_store: HistoryStore) -> Dict[str, float]:
    """
    Compute rolling-correlation-based adaptive weights.

    For each factor group, estimate its rolling correlation with forward
    returns across all tickers in the store.  Weight proportionally to
    max(0, correlation) so only factors that have been predictive recently
    get more weight.  Falls back to static weights if insufficient history.
    """
    factor_keys = ["technical", "fundamental", "sentiment", "structural", "setup"]
    return_key  = "upside"   # proxy: we use upside_score change as a forward return signal

    raw_corrs = {}
    for key in factor_keys:
        corr = history_store.cross_sectional_rolling_corr(
            factor_key=key,
            return_key=return_key,
            window=ADAPTIVE_WEIGHT_WINDOW,
        )
        raw_corrs[key] = max(0.0, corr)   # only reward positive predictors

    # If all correlations are near zero, fall back to static weights
    total = sum(raw_corrs.values())
    if total < 0.05:
        return dict(SCORING_WEIGHTS)

    # Normalize; keep regime as a fixed 10% overlay
    remaining = 0.90
    weights   = {}
    for k in factor_keys:
        weights[k] = remaining * (raw_corrs[k] / total)
    weights["regime"] = 0.10

    return weights


def compute_upside_score(
    tech_score: float,
    fund_score: float,
    sent_score: float,
    struct_score: float,
    setup_score: float,
    regime_score: float,
    history_store: HistoryStore,
) -> float:
    """
    Combine factor group scores into a single UpsideScore in [0, 1].

    Weights are adaptive (based on recent predictive power) when
    ADAPTIVE_WEIGHT_ENABLED=True, otherwise static from config.

    The regime_score acts as a multiplier gate:
      - regime_score < 0.3 → hard dampen the upside
      - regime_score > 0.7 → slight boost
    """
    if ADAPTIVE_WEIGHT_ENABLED:
        weights = _adaptive_weights(history_store)
    else:
        weights = dict(SCORING_WEIGHTS)

    scores = {
        "technical":   tech_score,
        "fundamental": fund_score,
        "sentiment":   sent_score,
        "structural":  struct_score,
        "setup":       setup_score,
        "regime":      regime_score,
    }

    upside = float(sum(weights.get(k, 0.0) * v for k, v in scores.items()))

    # Regime gate: dampen in bad regimes, minor boost in great ones
    if regime_score < 0.30:
        upside *= 0.70
    elif regime_score > 0.75:
        upside = float(np.clip(upside * 1.05, 0.0, 1.0))

    return float(np.clip(upside, 0.0, 1.0))
