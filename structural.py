# ─────────────────────────────────────────────
#  brains/structural.py  –  Structural brain
# ─────────────────────────────────────────────
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple

from history import HistoryStore


def compute_structural_factors(
    stock_data: Dict[str, Any],
    history_store: HistoryStore,
) -> Dict[str, float]:
    """Extract raw structural metrics."""
    si  = stock_data.get("short_interest", {})
    ins = stock_data.get("insider_activity", {})

    short_float  = float(si.get("short_float_pct", 5.0))
    days_cover   = float(si.get("days_to_cover", 2.0))
    borrow_cost  = float(si.get("borrow_cost_pct", 1.0))
    short_trend  = float(si.get("short_trend", 0.0))   # rising = shorts adding, falling = covering

    float_m      = float(stock_data.get("float_shares_m", 200.0))
    inst_own     = float(stock_data.get("inst_ownership_pct", 50.0))

    n_buys       = float(ins.get("buy_events_90d",  0))
    n_sells      = float(ins.get("sell_events_90d", 0))
    net_buy      = float(ins.get("net_buy_usd_90d", 0.0))
    largest_buy  = float(ins.get("largest_buy_usd", 0.0))

    # Price trend (used to assess squeeze pressure)
    prices = stock_data["prices"]
    price_trend = float((prices[-1] - prices[-20]) / (prices[-20] + 1e-9)) if len(prices) >= 20 else 0.0

    return {
        "short_float":   short_float,
        "days_to_cover": days_cover,
        "borrow_cost":   borrow_cost,
        "short_trend":   short_trend,
        "price_trend_20": price_trend,
        "float_m":        float_m,
        "inst_ownership": inst_own,
        "insider_buys":   n_buys,
        "insider_sells":  n_sells,
        "net_buy_usd":    net_buy,
        "largest_buy":    largest_buy,
    }


def score_structural_factors(
    raw_factors: Dict[str, float],
    history_store: HistoryStore,
) -> Tuple[float, Dict[str, float]]:
    """Convert structural metrics to 0–1 sub-scores."""

    # ── Short interest / squeeze score ────────────────────────────────────────
    sf   = raw_factors["short_float"]
    dtc  = raw_factors["days_to_cover"]
    bc   = raw_factors["borrow_cost"]
    st   = raw_factors["short_trend"]    # negative = shorts covering (good)
    pt   = raw_factors["price_trend_20"] # positive = price rising (squeezing shorts)

    # Short float: >20% very high, >10% high, <5% low
    if   sf > 25: sf_s = 1.00
    elif sf > 20: sf_s = 0.85
    elif sf > 15: sf_s = 0.70
    elif sf > 10: sf_s = 0.55
    elif sf > 5:  sf_s = 0.35
    else:         sf_s = 0.15

    # Days to cover
    if   dtc > 10: dtc_s = 1.00
    elif dtc > 7:  dtc_s = 0.80
    elif dtc > 5:  dtc_s = 0.65
    elif dtc > 3:  dtc_s = 0.45
    else:          dtc_s = 0.20

    # Borrow cost: higher = shorts under pressure
    bc_s = float(np.clip(bc / 20.0, 0.0, 1.0))

    # Short trend: falling shorts (covering) + rising price = squeeze momentum
    squeeze_momentum = float(np.clip(0.5 - st + pt, 0.0, 1.0))

    short_score = float(
        0.35 * sf_s +
        0.25 * dtc_s +
        0.15 * bc_s +
        0.25 * squeeze_momentum
    )

    # ── Float / ownership score ───────────────────────────────────────────────
    float_m  = raw_factors["float_m"]
    inst_own = raw_factors["inst_ownership"]

    # Small float = more explosive moves
    if   float_m < 10:   fl_s = 1.00
    elif float_m < 50:   fl_s = 0.80
    elif float_m < 200:  fl_s = 0.55
    elif float_m < 1000: fl_s = 0.35
    else:                fl_s = 0.15

    # High inst ownership: validates the name; very high can be a ceiling
    if   inst_own > 75: io_s = 0.70
    elif inst_own > 55: io_s = 0.85
    elif inst_own > 35: io_s = 0.65
    elif inst_own > 15: io_s = 0.45
    else:               io_s = 0.30

    float_score = float(0.60 * fl_s + 0.40 * io_s)

    # ── Insider score ─────────────────────────────────────────────────────────
    buys    = raw_factors["insider_buys"]
    sells   = raw_factors["insider_sells"]
    net     = raw_factors["net_buy_usd"]
    largest = raw_factors["largest_buy"]

    net_s = float(np.clip(0.5 + net / 1_000_000, 0.0, 1.0))   # $1M net buy → 1.0
    lg_s  = float(np.clip(largest / 500_000, 0.0, 1.0))         # $500k single buy → 1.0
    bal_s = float(np.clip(0.5 + (buys - sells) / 5.0, 0.0, 1.0))

    insider_score = float(0.40 * net_s + 0.30 * lg_s + 0.30 * bal_s)

    # ── Composite ─────────────────────────────────────────────────────────────
    sub_scores = {
        "short_squeeze": short_score,
        "float_ownership": float_score,
        "insider":        insider_score,
    }

    struct_score = float(
        0.45 * short_score +
        0.25 * float_score +
        0.30 * insider_score
    )

    return float(np.clip(struct_score, 0.0, 1.0)), sub_scores
