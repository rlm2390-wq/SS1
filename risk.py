# ─────────────────────────────────────────────
#  brains/risk.py  –  Risk brain
# ─────────────────────────────────────────────
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple

from storage.history import HistoryStore


def compute_risk_factors(
    stock_data: Dict[str, Any],
    history_store: HistoryStore,
) -> Dict[str, float]:
    """Extract raw risk metrics."""
    prices  = stock_data["prices"]
    volumes = stock_data["volumes"]
    atr_pct = float(stock_data.get("atr_pct", 0.02))
    adv     = float(stock_data.get("avg_dollar_vol", 1_000_000))

    # Gap risk: frequency of >5% gaps in last 60 days
    n = min(60, len(prices))
    daily_changes = np.abs(np.diff(prices[-n:]) / prices[-n:-1])
    gap_freq = float(np.mean(daily_changes > 0.05))

    # Realized volatility (annualized)
    ret_std = float(np.std(np.diff(prices[-60:]) / prices[-60:-1])) if len(prices) >= 61 else 0.02
    ann_vol = ret_std * np.sqrt(252)

    # Liquidity: average daily dollar volume
    avg_dv = float((prices[-20:] * volumes[-20:]).mean()) if len(prices) >= 20 else adv

    # Price itself (very low-priced stocks = higher risk)
    last_price = float(prices[-1])

    return {
        "atr_pct":   atr_pct,
        "gap_freq":  gap_freq,
        "ann_vol":   ann_vol,
        "avg_dv":    avg_dv,
        "last_price": last_price,
    }


def score_risk_factors(
    raw_factors: Dict[str, float],
    history_store: HistoryStore,
) -> Tuple[float, Dict[str, float]]:
    """
    Produce a RiskScore in [0, 1].  Higher = MORE risk.
    (Inverted convention: we filter out stocks with risk_score > threshold.)
    """

    # ── Liquidity risk ────────────────────────────────────────────────────────
    adv = raw_factors["avg_dv"]
    if   adv > 50_000_000:  liq_r = 0.05
    elif adv > 20_000_000:  liq_r = 0.15
    elif adv > 5_000_000:   liq_r = 0.30
    elif adv > 1_000_000:   liq_r = 0.55
    elif adv > 500_000:     liq_r = 0.75
    else:                   liq_r = 0.95

    # ── Volatility risk ───────────────────────────────────────────────────────
    atr = raw_factors["atr_pct"]
    if   atr < 0.01: vol_r = 0.10
    elif atr < 0.02: vol_r = 0.25
    elif atr < 0.04: vol_r = 0.45
    elif atr < 0.07: vol_r = 0.65
    elif atr < 0.12: vol_r = 0.80
    else:            vol_r = 0.95

    # ── Gap risk ──────────────────────────────────────────────────────────────
    gf   = raw_factors["gap_freq"]   # fraction of days with >5% gap
    gap_r = float(np.clip(gf / 0.20, 0.0, 1.0))   # 20% of days gapping → max risk

    # ── Price risk (penny stock penalty) ─────────────────────────────────────
    price = raw_factors["last_price"]
    if   price >= 10: price_r = 0.0
    elif price >= 5:  price_r = 0.3
    elif price >= 2:  price_r = 0.6
    else:             price_r = 0.9

    # ── Composite risk score ──────────────────────────────────────────────────
    sub_scores = {
        "liquidity": liq_r,
        "volatility": vol_r,
        "gap":        gap_r,
        "price":      price_r,
    }

    risk_score = float(
        0.40 * liq_r +
        0.35 * vol_r +
        0.15 * gap_r +
        0.10 * price_r
    )

    return float(np.clip(risk_score, 0.0, 1.0)), sub_scores
