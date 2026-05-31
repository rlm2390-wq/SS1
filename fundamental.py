# ─────────────────────────────────────────────
#  brains/fundamental.py  –  Fundamental brain
# ─────────────────────────────────────────────
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple

from storage.history import HistoryStore


def compute_fundamental_factors(
    stock_data: Dict[str, Any],
    sector_stats: Dict[str, Any],
    history_store: HistoryStore,
) -> Dict[str, float]:
    """
    Extract and lightly transform raw fundamental values.
    """
    f  = stock_data["fundamentals"]
    ss = sector_stats

    # Growth
    rev_yoy      = float(f.get("revenue_yoy", 0.0))
    eps_yoy      = float(f.get("eps_yoy", 0.0))
    margin_trend = float(f.get("margin_trend", 0.0))

    # Valuation
    peg          = float(f.get("peg", 2.0))
    ev_sales     = float(f.get("ev_sales", 3.0))
    sector_ev    = float(ss.get("ev_sales_median", 3.0))
    ev_vs_sector = ev_sales / max(sector_ev, 1e-9)   # <1 = cheap vs sector

    # Quality / balance sheet
    debt_equity     = float(f.get("debt_to_equity", 0.5))
    cash_runway     = float(f.get("cash_runway_years", 3.0))
    gross_margin    = float(f.get("gross_margin", 0.4))
    sect_margin     = float(ss.get("gross_margin_median", 0.4))
    margin_vs_sect  = gross_margin - sect_margin   # positive = above sector

    # Earnings quality
    earnings_surp   = float(f.get("earnings_surprise", 0.0))

    return {
        "rev_yoy":         rev_yoy,
        "eps_yoy":         eps_yoy,
        "margin_trend":    margin_trend,
        "peg":             peg,
        "ev_vs_sector":    ev_vs_sector,
        "debt_equity":     debt_equity,
        "cash_runway":     cash_runway,
        "margin_vs_sect":  margin_vs_sect,
        "earnings_surp":   earnings_surp,
    }


def score_fundamental_factors(
    raw_factors: Dict[str, float],
    history_store: HistoryStore,
) -> Tuple[float, Dict[str, float]]:
    """
    Convert raw fundamental metrics to 0–1 sub-scores, then blend.
    Uses cross-sectional percentile ranks from history when available.
    """

    # ── Growth score ──────────────────────────────────────────────────────────
    rv  = raw_factors["rev_yoy"]
    eps = raw_factors["eps_yoy"]
    mt  = raw_factors["margin_trend"]

    if   rv > 0.30: rv_s = 1.00
    elif rv > 0.20: rv_s = 0.85
    elif rv > 0.10: rv_s = 0.65
    elif rv > 0.00: rv_s = 0.45
    else:           rv_s = 0.15

    if   eps > 0.30: eps_s = 1.00
    elif eps > 0.20: eps_s = 0.85
    elif eps > 0.10: eps_s = 0.65
    elif eps > 0.00: eps_s = 0.45
    else:            eps_s = 0.15

    margin_s = float(np.clip(0.5 + mt / 0.10, 0.0, 1.0))   # ±10% swing maps to 0–1

    growth_score = float(0.40 * rv_s + 0.40 * eps_s + 0.20 * margin_s)

    # ── Valuation score ───────────────────────────────────────────────────────
    peg = raw_factors["peg"]
    evs = raw_factors["ev_vs_sector"]

    # PEG: <1 great, 1-2 ok, >2 stretched
    if   peg < 0.8: peg_s = 1.00
    elif peg < 1.2: peg_s = 0.80
    elif peg < 2.0: peg_s = 0.55
    elif peg < 3.0: peg_s = 0.30
    else:           peg_s = 0.10

    # EV/S vs sector: <0.8x = undervalued, >1.5x = overvalued
    if   evs < 0.60: evs_s = 1.00
    elif evs < 0.80: evs_s = 0.80
    elif evs < 1.20: evs_s = 0.55
    elif evs < 1.50: evs_s = 0.30
    else:            evs_s = 0.10

    valuation_score = float(0.50 * peg_s + 0.50 * evs_s)

    # ── Balance sheet / quality score ─────────────────────────────────────────
    de   = raw_factors["debt_equity"]
    cash = raw_factors["cash_runway"]
    marg = raw_factors["margin_vs_sect"]
    surp = raw_factors["earnings_surp"]

    de_s   = float(np.clip(1.0 - de / 3.0, 0.0, 1.0))         # 0 debt → 1.0
    cash_s = float(np.clip((cash - 0.5) / 4.0, 0.0, 1.0))      # >4.5yr runway → 1.0
    marg_s = float(np.clip(0.5 + marg / 0.20, 0.0, 1.0))       # +20pp above sector → 1.0
    surp_s = float(np.clip(0.5 + surp / 0.10, 0.0, 1.0))       # +10% beat → 1.0

    quality_score = float(0.30 * de_s + 0.30 * cash_s + 0.25 * marg_s + 0.15 * surp_s)

    # ── Composite ─────────────────────────────────────────────────────────────
    sub_scores = {
        "growth":    growth_score,
        "valuation": valuation_score,
        "quality":   quality_score,
    }

    fund_score = float(
        0.45 * growth_score +
        0.30 * valuation_score +
        0.25 * quality_score
    )

    return float(np.clip(fund_score, 0.0, 1.0)), sub_scores
