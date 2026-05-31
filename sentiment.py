# ─────────────────────────────────────────────
#  brains/sentiment.py  –  Sentiment / flow brain
# ─────────────────────────────────────────────
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple

from storage.history import HistoryStore


def compute_sentiment_factors(
    stock_data: Dict[str, Any],
    history_store: HistoryStore,
) -> Dict[str, float]:
    """
    Compute raw sentiment and options-flow metrics.
    """
    news_items   = stock_data.get("news_items", [])
    options_flow = stock_data.get("options_flow", {})

    # ── News sentiment ────────────────────────────────────────────────────────
    if news_items:
        sentiments      = [n["sentiment"] for n in news_items]
        avg_sentiment   = float(np.mean(sentiments))
        news_count_7d   = len(news_items)
        sentiment_stdev = float(np.std(sentiments)) if len(sentiments) > 1 else 0.0
    else:
        avg_sentiment   = 0.0
        news_count_7d   = 0
        sentiment_stdev = 0.0

    # ── Analyst revisions ─────────────────────────────────────────────────────
    upgrades   = int(stock_data.get("analyst_upgrades_30d",   0))
    downgrades = int(stock_data.get("analyst_downgrades_30d", 0))
    tgt_chg    = float(stock_data.get("analyst_target_change_pct", 0.0))

    # ── Options / flow ────────────────────────────────────────────────────────
    call_vol_ratio  = float(options_flow.get("call_vol_ratio",  1.0))
    call_put_ratio  = float(options_flow.get("call_put_ratio",  1.0))
    iv_percentile   = float(options_flow.get("iv_percentile",   50.0))
    unusual_options = bool(options_flow.get("unusual_options",  False))

    return {
        "avg_sentiment":    avg_sentiment,
        "news_count_7d":    float(news_count_7d),
        "sentiment_stdev":  sentiment_stdev,
        "analyst_upgrades": float(upgrades),
        "analyst_downs":    float(downgrades),
        "target_change":    tgt_chg,
        "call_vol_ratio":   call_vol_ratio,
        "call_put_ratio":   call_put_ratio,
        "iv_percentile":    iv_percentile,
        "unusual_options":  float(unusual_options),
    }


def score_sentiment_factors(
    raw_factors: Dict[str, float],
    history_store: HistoryStore,
) -> Tuple[float, Dict[str, float]]:
    """Normalize sentiment metrics to 0–1 sub-scores and blend."""

    # ── News sentiment score ──────────────────────────────────────────────────
    sent   = raw_factors["avg_sentiment"]      # -1 to +1
    count  = raw_factors["news_count_7d"]
    stdev  = raw_factors["sentiment_stdev"]

    # Map -1..+1 → 0..1
    base_news = float((sent + 1.0) / 2.0)

    # Low news count → less confidence; penalize mixed signals (high stdev)
    confidence = float(np.clip(count / 5.0, 0.3, 1.0))
    consensus  = float(np.clip(1.0 - stdev, 0.0, 1.0))

    news_score = float(base_news * confidence * (0.5 + 0.5 * consensus))
    news_score = float(np.clip(news_score, 0.0, 1.0))

    # ── Analyst score ─────────────────────────────────────────────────────────
    ups  = raw_factors["analyst_upgrades"]
    dns  = raw_factors["analyst_downs"]
    tgt  = raw_factors["target_change"]

    net_revs = ups - dns
    rev_s    = float(np.clip(0.5 + net_revs / 3.0, 0.0, 1.0))
    tgt_s    = float(np.clip(0.5 + tgt / 0.20, 0.0, 1.0))     # ±20% target change → 0..1

    analyst_score = float(0.50 * rev_s + 0.50 * tgt_s)

    # ── Options / flow score ──────────────────────────────────────────────────
    cvr = raw_factors["call_vol_ratio"]    # >1 = elevated call buying
    cpr = raw_factors["call_put_ratio"]    # >1 = more calls than puts
    iv  = raw_factors["iv_percentile"]     # 0-100
    uno = raw_factors["unusual_options"]   # 0 or 1

    cvr_s = float(np.clip((cvr - 1.0) / 2.0 + 0.5, 0.0, 1.0))   # 1x→0.5, 3x→1.0
    cpr_s = float(np.clip((cpr - 0.5) / 2.0, 0.0, 1.0))

    # IV percentile: mid-range IV (30-60) is ideal for long calls; very high IV is expensive
    if   iv < 20:  iv_s = 0.70
    elif iv < 40:  iv_s = 0.90
    elif iv < 60:  iv_s = 0.70
    elif iv < 80:  iv_s = 0.45
    else:          iv_s = 0.20
    iv_s = float(iv_s)

    flow_score = float(
        0.35 * cvr_s +
        0.30 * cpr_s +
        0.20 * iv_s +
        0.15 * float(uno)
    )

    # ── Composite ─────────────────────────────────────────────────────────────
    sub_scores = {
        "news":     news_score,
        "analyst":  analyst_score,
        "flow":     flow_score,
    }

    sent_score = float(
        0.35 * news_score +
        0.35 * analyst_score +
        0.30 * flow_score
    )

    return float(np.clip(sent_score, 0.0, 1.0)), sub_scores
