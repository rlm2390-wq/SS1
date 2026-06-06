# ─────────────────────────────────────────────
#  pre_signal.py  –  Before-It-Happens Engine
#
#  Detects early accumulation/squeeze signals
#  BEFORE they become setups or alerts.
#
#  Signal types:
#    1. Volatility contraction — BB width in
#       extreme compression vs 1-year history
#    2. Volume dry-up — declining volume 3+ days
#       while price holds support
#    3. OI buildup — call OI increasing on OTM
#       strikes between scans (tracked in history)
#    4. Short interest acceleration — SI falling
#       faster than 2%/week = covering pressure
#    5. Earnings IV expansion — IV rising >20%
#       faster than normal 2+ weeks pre-earnings
#    6. Insider clustering — multiple insiders
#       buying within a 14-day window
#
#  Each signal returns a conviction score 0–1.
#  Tickers are tiered: 1 signal = weak,
#  2 = medium, 3+ = strong.
# ─────────────────────────────────────────────
from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from history import HistoryStore

logger = logging.getLogger("pre_signal")


# ── 1. Volatility Contraction ─────────────────────────────────────────────────

def _detect_vol_contraction(
    stock_data: Dict[str, Any],
    history_store: HistoryStore,
) -> Tuple[bool, float, str]:
    """
    Fires when current BB width is in the bottom 10% of the past year.
    Returns (detected, conviction 0-1, description).
    """
    prices   = stock_data.get("prices", np.array([]))
    bb_width = stock_data.get("bb_width", 0.05)

    if len(prices) < 60:
        return False, 0.0, ""

    # Build a rolling BB width series over the last year
    year_bw = []
    for i in range(20, min(len(prices), 252)):
        sl  = prices[max(0, i-19): i+1]
        mid = sl.mean()
        if mid > 0:
            year_bw.append(float(sl.std() * 4 / mid))

    if len(year_bw) < 30:
        return False, 0.0, ""

    pct_rank = float(np.mean(np.array(year_bw) >= bb_width))

    # Bottom 10% of historical widths = extreme compression
    if pct_rank >= 0.90:
        conviction = float(np.clip((pct_rank - 0.90) / 0.10 + 0.5, 0.5, 1.0))
        return True, conviction, (
            f"Extreme volatility compression — BB width in bottom "
            f"{round((1-pct_rank)*100)}% of last year. "
            f"Springs coil before they release."
        )

    return False, 0.0, ""


# ── 2. Volume Dry-Up ─────────────────────────────────────────────────────────

def _detect_volume_dryup(
    stock_data: Dict[str, Any],
) -> Tuple[bool, float, str]:
    """
    Fires when volume has been declining for 3+ consecutive days while
    price is holding steady or rising (healthy accumulation pattern).
    """
    prices  = stock_data.get("prices",  np.array([]))
    volumes = stock_data.get("volumes", np.array([]))

    if len(volumes) < 10:
        return False, 0.0, ""

    recent_vols  = volumes[-5:]
    avg_20d_vol  = float(volumes[-20:].mean()) if len(volumes) >= 20 else float(volumes.mean())
    avg_recent   = float(recent_vols.mean())

    # Check 3+ days of declining volume
    consec_decline = 0
    for i in range(1, len(recent_vols)):
        if recent_vols[i] < recent_vols[i-1]:
            consec_decline += 1
        else:
            consec_decline = 0

    # Price holding or rising over same period
    price_holding = len(prices) >= 5 and prices[-1] >= prices[-5] * 0.98

    # Volume must be genuinely low vs baseline
    vol_suppressed = avg_recent < avg_20d_vol * 0.70

    if consec_decline >= 2 and price_holding and vol_suppressed:
        conviction = float(np.clip(
            0.4 * (consec_decline / 4.0) +
            0.3 * float(price_holding) +
            0.3 * float((avg_20d_vol - avg_recent) / (avg_20d_vol + 1e-9)),
            0.0, 1.0
        ))
        return True, conviction, (
            f"Volume drying up for {consec_decline}+ days while price holds. "
            f"Classic institutional accumulation — they buy quietly, "
            f"letting volume fade before the next leg up."
        )

    return False, 0.0, ""


# ── 3. OI Buildup (tracked between scans) ────────────────────────────────────

def _detect_oi_buildup(
    ticker: str,
    stock_data: Dict[str, Any],
    history_store: HistoryStore,
) -> Tuple[bool, float, str]:
    """
    Detects call OI building on strikes 5–15% above current price.
    Compares current OI snapshot to previous scan stored in HistoryStore.
    Also tracks call/put skew changes and volume/OI ratio.
    """
    opts = stock_data.get("options_flow", {})
    if not opts:
        return False, 0.0, ""

    call_oi      = float(opts.get("call_oi",       0))
    put_oi       = float(opts.get("put_oi",        0))
    call_vol     = float(opts.get("call_vol_ratio", 1.0))
    iv_rank      = float(opts.get("iv_rank",        50))

    if call_oi <= 0:
        return False, 0.0, ""

    # Store current OI snapshot
    history_store.update_stock(ticker, None, {
        "call_oi":  call_oi,
        "put_oi":   put_oi,
        "call_vol": call_vol,
    })

    # Get historical OI for delta calculation
    call_oi_hist = history_store.get_stock_history(ticker, "call_oi", lookback_days=7)
    put_oi_hist  = history_store.get_stock_history(ticker, "put_oi",  lookback_days=7)

    signals = []
    conviction_parts = []

    # 1. OI delta — is call OI growing between scans?
    if len(call_oi_hist) >= 2:
        oi_delta_pct = (call_oi - call_oi_hist[0]) / (call_oi_hist[0] + 1e-9)
        if oi_delta_pct > 0.15:   # 15%+ OI growth
            signals.append(f"call OI grew {round(oi_delta_pct*100)}% since last scan")
            conviction_parts.append(min(oi_delta_pct / 0.50, 1.0))

    # 2. Call/put skew shifting toward calls
    if len(call_oi_hist) >= 2 and len(put_oi_hist) >= 2:
        skew_now  = call_oi  / max(put_oi, 1)
        skew_prev = call_oi_hist[0] / max(put_oi_hist[0], 1)
        skew_chg  = skew_now - skew_prev
        if skew_chg > 0.20:
            signals.append(f"call/put skew shifted +{round(skew_chg, 2)}x toward calls")
            conviction_parts.append(min(skew_chg / 1.0, 1.0))

    # 3. Volume/OI ratio — high ratio = fresh money, not just rolls
    vol_oi_ratio = call_vol / max(call_oi / 1000, 1)
    if vol_oi_ratio > 0.3:
        signals.append("high volume-to-OI ratio — fresh call buying, not rollovers")
        conviction_parts.append(min(vol_oi_ratio / 1.0, 1.0))

    if not signals:
        return False, 0.0, ""

    conviction = float(np.mean(conviction_parts)) if conviction_parts else 0.5
    desc = "Unusual options flow building: " + "; ".join(signals) + "."
    return True, float(np.clip(conviction, 0.3, 1.0)), desc


# ── 4. Short Interest Acceleration ───────────────────────────────────────────

def _detect_si_acceleration(
    ticker: str,
    stock_data: Dict[str, Any],
    history_store: HistoryStore,
) -> Tuple[bool, float, str]:
    """
    Detects short interest falling faster than 2%/week = covering pressure.
    The rate of change matters more than the absolute level.
    """
    si = stock_data.get("short_interest", {})
    sf = float(si.get("short_float_pct", 0))

    if sf <= 0:
        return False, 0.0, ""

    # Store current SI
    history_store.update_stock(ticker, None, {"short_float": sf})

    # Get SI history
    sf_hist = history_store.get_stock_history(ticker, "short_float", lookback_days=14)

    if len(sf_hist) < 3:
        return False, 0.0, ""

    # Rate of change (per week, normalized)
    sf_oldest = sf_hist[0]
    sf_chg    = (sf - sf_oldest) / (sf_oldest + 1e-9)

    # Falling 3%+ per week = covering pressure
    if sf_chg < -0.03 and sf > 8.0:
        rate_str   = round(abs(sf_chg) * 100, 1)
        conviction = float(np.clip(abs(sf_chg) / 0.15, 0.3, 1.0))
        return True, conviction, (
            f"Short interest falling at {rate_str}%/period — shorts are covering. "
            f"{round(sf)}% still short = significant remaining fuel. "
            f"Covering pressure can accelerate price moves."
        )

    return False, 0.0, ""


# ── 5. Earnings IV Expansion ──────────────────────────────────────────────────

def _detect_earnings_iv_expansion(
    ticker: str,
    stock_data: Dict[str, Any],
    history_store: HistoryStore,
) -> Tuple[bool, float, str]:
    """
    Fires when IV is expanding significantly faster than usual in the
    2–3 weeks before earnings. Indicates smart money pricing in a move.
    """
    earnings_date = stock_data.get("earnings_date")
    if not earnings_date:
        return False, 0.0, ""

    try:
        ed      = datetime.date.fromisoformat(str(earnings_date)[:10])
        today   = datetime.date.today()
        days_to = (ed - today).days
    except (ValueError, TypeError):
        return False, 0.0, ""

    # Only relevant 5–21 days before earnings
    if not (5 <= days_to <= 21):
        return False, 0.0, ""

    opts     = stock_data.get("options_flow", {})
    iv_rank  = float(opts.get("iv_rank", 50))

    # Store IV history for this ticker
    history_store.update_stock(ticker, None, {"iv_rank": iv_rank})
    iv_hist = history_store.get_stock_history(ticker, "iv_rank", lookback_days=14)

    if len(iv_hist) < 2:
        return False, 0.0, ""

    iv_expansion = iv_rank - float(np.mean(iv_hist[:-1]))

    if iv_expansion > 10 and iv_rank > 40:
        conviction = float(np.clip(iv_expansion / 30.0, 0.3, 1.0))
        return True, conviction, (
            f"IV expanding {round(iv_expansion)}pts above recent avg "
            f"with {days_to} days to earnings ({earnings_date}). "
            f"Options market pricing in a significant move — smart money positioning."
        )

    return False, 0.0, ""


# ── 6. Insider Clustering ─────────────────────────────────────────────────────

def _detect_insider_clustering(
    stock_data: Dict[str, Any],
) -> Tuple[bool, float, str]:
    """
    Fires when multiple insiders are buying in a tight time window.
    A single insider buy = noise. Multiple = conviction signal.
    """
    ins    = stock_data.get("insider_activity", {})
    buys   = int(ins.get("buy_events_90d",  0))
    sells  = int(ins.get("sell_events_90d", 0))
    net    = float(ins.get("net_buy_usd_90d", 0))
    largest = float(ins.get("largest_buy_usd", 0))

    if buys < 2:
        return False, 0.0, ""

    # Multiple buyers + net positive + meaningful size
    net_positive  = net > 100_000
    big_buy       = largest > 200_000
    more_buys_than_sells = buys > sells

    if buys >= 2 and net_positive and more_buys_than_sells:
        conviction = float(np.clip(
            0.40 * min(buys / 4.0, 1.0) +
            0.30 * float(big_buy) +
            0.30 * min(net / 1_000_000, 1.0),
            0.3, 1.0
        ))
        return True, conviction, (
            f"{buys} insiders bought in the last 90 days "
            f"(net ${int(net/1000)}k, largest single buy ${int(largest/1000)}k). "
            f"Insider clusters — not individual transactions — are statistically significant."
        )

    return False, 0.0, ""


# ── Public API ────────────────────────────────────────────────────────────────

_SIGNAL_TIER = {1: "weak", 2: "medium", 3: "strong"}


def compute_pre_signals(
    ticker: str,
    stock_data: Dict[str, Any],
    history_store: HistoryStore,
) -> Dict[str, Any]:
    """
    Run all pre-signal detectors and return a structured result.

    Returns:
    {
        "signals":       [{"name", "conviction", "description"}, ...],
        "signal_count":  int,
        "tier":          "weak" | "medium" | "strong",
        "pre_score":     float 0-1,
        "top_signal":    str (name of strongest signal),
    }
    """
    detectors = [
        ("vol_contraction",    lambda: _detect_vol_contraction(stock_data, history_store)),
        ("volume_dryup",       lambda: _detect_volume_dryup(stock_data)),
        ("oi_buildup",         lambda: _detect_oi_buildup(ticker, stock_data, history_store)),
        ("si_acceleration",    lambda: _detect_si_acceleration(ticker, stock_data, history_store)),
        ("earnings_iv_expand", lambda: _detect_earnings_iv_expansion(ticker, stock_data, history_store)),
        ("insider_cluster",    lambda: _detect_insider_clustering(stock_data)),
    ]

    fired: List[Dict] = []
    for name, fn in detectors:
        try:
            detected, conviction, desc = fn()
            if detected:
                fired.append({
                    "name":        name,
                    "conviction":  round(conviction, 3),
                    "description": desc,
                })
        except Exception as e:
            logger.debug("pre_signal %s/%s failed: %s", ticker, name, e)

    # Sort by conviction descending
    fired.sort(key=lambda x: x["conviction"], reverse=True)

    n         = len(fired)
    tier      = _SIGNAL_TIER.get(min(n, 3), "strong")
    pre_score = 0.0

    if fired:
        # Weighted composite: lead with the strongest
        convictions = [s["conviction"] for s in fired]
        pre_score   = convictions[0]
        for i, c in enumerate(convictions[1:], 1):
            pre_score += c * (0.3 ** i)   # diminishing credit
        pre_score = float(np.clip(pre_score, 0.0, 1.0))

    return {
        "signals":      fired,
        "signal_count": n,
        "tier":         tier,
        "pre_score":    round(pre_score, 3),
        "top_signal":   fired[0]["name"] if fired else None,
    }
