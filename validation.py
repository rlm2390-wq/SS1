# ─────────────────────────────────────────────
#  brains/validation.py  –  Data sanity brain
# ─────────────────────────────────────────────
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Tuple


def validate_data(stock_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Run sanity checks on stock_data before any brain processes it.

    Returns
    -------
    (is_valid, issues)
        is_valid : False means skip this ticker entirely.
        issues   : Human-readable list of problems found.
    """
    issues: List[str] = []

    # ── Required keys ─────────────────────────────────────────────────────────
    required = ["prices", "volumes", "ma20", "ma50", "fundamentals"]
    for k in required:
        if k not in stock_data:
            issues.append(f"Missing required field: {k}")

    if issues:
        return False, issues   # can't continue without basics

    prices  = stock_data["prices"]
    volumes = stock_data["volumes"]

    # ── Length checks ─────────────────────────────────────────────────────────
    if len(prices) < 20:
        issues.append(f"Insufficient price history: {len(prices)} bars (need 20+)")
        return False, issues

    if len(prices) != len(volumes):
        issues.append(f"Price/volume length mismatch: {len(prices)} vs {len(volumes)}")
        return False, issues

    # ── Price sanity ──────────────────────────────────────────────────────────
    if np.any(prices <= 0):
        issues.append("Non-positive prices detected")
        return False, issues

    if np.any(~np.isfinite(prices)):
        issues.append("Non-finite prices (NaN or Inf)")
        return False, issues

    last = float(prices[-1])
    if last < 0.10:
        issues.append(f"Price too low: ${last:.4f} (likely delisted or halted)")
        # Warn but don't block – let UNIVERSE_CONFIG filter handle it
        # return False, issues

    # ── Outlier check: single-bar returns ─────────────────────────────────────
    rets = np.diff(prices) / prices[:-1]
    extreme = np.abs(rets) > 0.50   # >50% single-bar move
    if extreme.sum() > 3:
        issues.append(f"{extreme.sum()} bars with >50% single-day return – possible data error")

    # ── Volume sanity ─────────────────────────────────────────────────────────
    if np.any(volumes < 0):
        issues.append("Negative volume detected")
        return False, issues

    if float(volumes[-20:].mean()) < 1000:
        issues.append("Average daily volume < 1,000 shares – extremely illiquid")

    # ── Fundamentals spot checks ──────────────────────────────────────────────
    f = stock_data.get("fundamentals", {})
    if "ev_sales" in f and (f["ev_sales"] < 0 or f["ev_sales"] > 500):
        issues.append(f"EV/Sales out of range: {f['ev_sales']:.1f}")

    if "debt_to_equity" in f and f["debt_to_equity"] < 0:
        issues.append("Negative debt/equity – check data")

    is_valid = len([i for i in issues if "Missing" in i or "mismatch" in i
                    or "Non-positive" in i or "Non-finite" in i
                    or "Negative volume" in i]) == 0

    return is_valid, issues
