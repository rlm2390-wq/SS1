# ─────────────────────────────────────────────
#  validation.py  –  Data sanity checks
#  Lenient enough to work on weekends /
#  after hours when some fields are null.
# ─────────────────────────────────────────────
from __future__ import annotations
import logging
import numpy as np
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("validation")


def validate_data(stock_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Run sanity checks on stock_data before any brain processes it.
    Returns (is_valid, issues).
    is_valid=False means skip this ticker entirely.

    Deliberately lenient — real-time quote fields (preMarketPrice,
    postMarketPrice) are often null on weekends/after-hours and
    should NOT cause a ticker to be rejected.
    """
    issues: List[str] = []

    # ── Required structural keys ──────────────────────────────────────────────
    # Only fail on things the brains absolutely cannot work without
    for key in ("prices", "volumes", "ma20", "ma50"):
        if key not in stock_data or stock_data[key] is None:
            issues.append(f"Missing required field: {key}")

    if issues:
        return False, issues

    prices  = stock_data["prices"]
    volumes = stock_data["volumes"]

    # ── Minimum history (lowered to 10 for weekend/holiday tolerance) ─────────
    if len(prices) < 10:
        issues.append(f"Insufficient price history: {len(prices)} bars (need 10+)")
        return False, issues

    # ── Length consistency ────────────────────────────────────────────────────
    if len(prices) != len(volumes):
        issues.append(f"Price/volume mismatch: {len(prices)} vs {len(volumes)}")
        # Try to fix by truncating rather than rejecting
        min_len = min(len(prices), len(volumes))
        prices  = prices[-min_len:]
        volumes = volumes[-min_len:]

    # ── Price sanity ──────────────────────────────────────────────────────────
    if np.any(prices <= 0):
        issues.append("Non-positive prices detected")
        return False, issues

    if np.any(~np.isfinite(prices)):
        issues.append("Non-finite prices (NaN/Inf)")
        return False, issues

    last = float(prices[-1])
    if last < 0.10:
        # Warn only — don't reject, let universe config handle price filtering
        issues.append(f"Price very low: ${last:.4f}")

    # ── Volume sanity (warn only, don't reject) ───────────────────────────────
    if len(volumes) >= 20:
        avg_vol = float(volumes[-20:].mean())
        if avg_vol < 100:
            issues.append(f"Very low avg volume: {avg_vol:.0f} shares")
            # Still process — let liquidity risk score handle it

    # ── Fundamentals (optional — warn only) ──────────────────────────────────
    f = stock_data.get("fundamentals", {})
    if f and "ev_sales" in f:
        if f["ev_sales"] < 0 or f["ev_sales"] > 500:
            issues.append(f"EV/Sales out of range: {f['ev_sales']:.1f}")

    # Only hard-fail on structural data problems, not optional field issues
    hard_fails = [i for i in issues if any(x in i for x in
        ("Missing required", "Insufficient", "Non-positive", "Non-finite"))]

    is_valid = len(hard_fails) == 0

    if not is_valid:
        logger.debug("Validation failed: %s", "; ".join(hard_fails))

    return is_valid, issues
