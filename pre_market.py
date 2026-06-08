# ─────────────────────────────────────────────
#  pre_market.py  –  Pre-Market Alerter
#
#  Scans the FULL universe independently.
#  Does NOT depend on main scan results,
#  watchlist, or alert history to run.
#
#  Tiering is a filter applied after scanning,
#  not a prerequisite for scanning.
#
#  Data source is abstracted via data_router
#  so switching to Webull real-time is trivial.
#
#  Tier 1 — Watchlist + Recent Alerts
#    Always fire on ≥1.5% move.
#    Shown in "🔥 Hot Pre-Market".
#
#  Tier 2 — Full Universe
#    Fire on ≥1.0% move if qualifying signals.
#    Shown in "🌅 Pre-Market Discovery".
# ─────────────────────────────────────────────
from __future__ import annotations

import datetime
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

from config import (
    REQUEST_DELAY_SECONDS, REQUEST_MAX_RETRIES, REQUEST_RETRY_BACKOFF,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
)
import cache as _cache

logger = logging.getLogger("pre_market")

# ── Timing ────────────────────────────────────────────────────────────────────
_PRE_MARKET_START_ET = datetime.time(4,  0)
_PRE_MARKET_END_ET   = datetime.time(9, 30)
_SCAN_INTERVAL_S     = 300   # 5 minutes

# ── Thresholds ────────────────────────────────────────────────────────────────
_TIER1_MOVE_PCT  = 0.015   # 1.5%
_TIER2_MOVE_PCT  = 0.010   # 1.0%

# ── State ─────────────────────────────────────────────────────────────────────
_pm_lock             = threading.Lock()
_pm_thread: Optional[threading.Thread] = None
_pm_running          = False
_hot_alerts:       List[Dict] = []
_discovery_alerts: List[Dict] = []
_last_pm_scan:     Optional[str] = None
_pm_scan_count     = 0
_fired_keys:       Set[str] = set()


# ── Pre-market universe ───────────────────────────────────────────────────────
# Independent of main scan — always has something to scan

def _get_pm_universe() -> List[str]:
    """
    Build the pre-market scan universe without depending on main scan.
    Reads from cache if available, otherwise fetches fresh.
    """
    tickers: List[str] = []

    # S&P 500 top 100 (cached for 24h)
    try:
        from universe import get_sp500_tickers
        sp500 = get_sp500_tickers()
        tickers.extend(sp500[:100])
    except Exception as e:
        logger.debug("PM universe: sp500 fetch error: %s", e)
        from universe import _SP500_FALLBACK
        tickers.extend(_SP500_FALLBACK[:50])

    # Deduplicate
    return list(dict.fromkeys(t.upper() for t in tickers))


def _get_tier1_tickers(
    watchlist:     List[str],
    recent_alerts: List[str],
    pm_universe:   List[str],
) -> List[str]:
    """
    Tier 1 = watchlist + recent alerts.
    If both are empty, fall back to top 20 from universe
    so we always have something to scan at tier 1.
    """
    tier1 = list(dict.fromkeys(
        [t.upper() for t in (watchlist or [])] +
        [t.upper() for t in (recent_alerts or [])]
    ))
    # Always have at least the top universe tickers at tier 1
    if len(tier1) < 5:
        tier1 = list(dict.fromkeys(tier1 + pm_universe[:20]))
    return tier1


# ── ET timezone helper ────────────────────────────────────────────────────────

def _now_et() -> datetime.datetime:
    utc_now = datetime.datetime.utcnow()
    month   = utc_now.month
    is_edt  = 3 <= month <= 11
    offset  = -4 if is_edt else -5
    return utc_now + datetime.timedelta(hours=offset)


def is_premarket() -> bool:
    now_et = _now_et().time()
    return _PRE_MARKET_START_ET <= now_et < _PRE_MARKET_END_ET


def premarket_status() -> Dict:
    now_et = _now_et()
    active = is_premarket()
    start  = now_et.replace(
        hour=_PRE_MARKET_START_ET.hour,
        minute=_PRE_MARKET_START_ET.minute, second=0, microsecond=0)
    end    = now_et.replace(
        hour=_PRE_MARKET_END_ET.hour,
        minute=_PRE_MARKET_END_ET.minute, second=0, microsecond=0)

    if active:
        mins_left  = int((end - now_et).total_seconds() / 60)
        status_str = f"Active — {mins_left}m until market open"
    else:
        if now_et.time() < _PRE_MARKET_START_ET:
            mins_to    = int((start - now_et).total_seconds() / 60)
            status_str = f"Starts in {mins_to}m (4:00 AM ET)"
        else:
            status_str = "Closed — market hours active"

    return {
        "active":      active,
        "status":      status_str,
        "et_time":     now_et.strftime("%I:%M %p ET"),
        "scan_count":  _pm_scan_count,
        "last_scan":   _last_pm_scan,
    }


# ── Data source abstraction ───────────────────────────────────────────────────
# Switching to Webull: replace _fetch_pm_data with data_router.get_quotes_batch

def _fetch_pm_data(tickers: List[str]) -> Dict[str, Dict]:
    """
    Fetch pre-market price data for a list of tickers.
    Currently uses yfinance. Webull-ready: swap for data_router.get_quotes_batch.
    """
    # Check cache first
    results: Dict[str, Dict] = {}
    uncached = []
    for t in tickers:
        cached = _cache.get_quote(t)
        if cached and cached.get("pre_market_price"):
            results[t] = cached
        else:
            uncached.append(t)

    if not uncached:
        return results

    # Future Webull path (one line to enable):
    # from data_router import get_quotes_batch
    # fresh = get_quotes_batch(uncached)
    # results.update(fresh); return results

    # yfinance path
    if not YF_AVAILABLE:
        return results

    for ticker in uncached:
        delay = REQUEST_DELAY_SECONDS
        for attempt in range(REQUEST_MAX_RETRIES):
            try:
                time.sleep(delay)
                info = yf.Ticker(ticker).info or {}

                prev_close = float(info.get("previousClose") or
                                   info.get("regularMarketPreviousClose") or 0)
                pm_price   = float(info.get("preMarketPrice")  or 0)
                pm_chg_pct = float(info.get("preMarketChangePercent") or
                                   (((pm_price - prev_close) / prev_close)
                                    if prev_close else 0))
                pm_vol     = int(info.get("preMarketVolume")   or 0)
                reg_price  = float(info.get("regularMarketPrice") or
                                   info.get("currentPrice")   or 0)

                if pm_price > 0:
                    data = {
                        "ticker":           ticker,
                        "available":        True,
                        "pre_market_price": pm_price,
                        "prev_close":       prev_close,
                        "change_pct":       pm_chg_pct / 100 if abs(pm_chg_pct) > 1 else pm_chg_pct,
                        "change_abs":       pm_price - prev_close,
                        "pm_volume":        pm_vol,
                        "reg_price":        reg_price,
                    }
                    results[ticker] = data
                    _cache.cache_quote(ticker, data)
                break

            except Exception as e:
                msg = str(e).lower()
                if "too many" in msg or "429" in msg or "rate" in msg:
                    wait = delay * (REQUEST_RETRY_BACKOFF ** attempt)
                    time.sleep(wait)
                    delay = wait
                else:
                    logger.debug("PM fetch failed %s: %s", ticker, e)
                    break

    return results


# ── Signal tier qualification ─────────────────────────────────────────────────

def _tier2_qualifies(result: Optional[Dict]) -> bool:
    """
    Optional qualification for tier 2 discovery.
    If result is None (ticker not in scan results), still qualify —
    pre-market scans the full universe regardless.
    """
    if result is None:
        return True   # Unknown ticker → always scan
    ps = result.get("pre_signals", {})
    return (
        ps.get("signal_count", 0) >= 2 or
        bool(set(result.get("setups", [])) & {
            "short_squeeze", "volatility_breakout",
            "trend_pullback", "earnings_drift"}) or
        result.get("microstructure", {}).get("microstructure_score", 0) >= 0.55 or
        result.get("upside", 0) >= 0.65
    )


# ── Alert builder ─────────────────────────────────────────────────────────────

def _build_alert(
    ticker:     str,
    pm_data:    Dict,
    tier:       int,
    result:     Optional[Dict],
    reason:     str,
) -> Dict:
    chg_pct = pm_data.get("change_pct", 0)
    # Sanity cap — yfinance sometimes returns garbage
    chg_pct = max(-0.50, min(0.50, chg_pct))
    return {
        "ticker":       ticker,
        "tier":         tier,
        "tier_label":   "🔥 Hot" if tier == 1 else "🌅 Discovery",
        "pm_price":     round(pm_data.get("pre_market_price", 0), 2),
        "prev_close":   round(pm_data.get("prev_close",       0), 2),
        "change_pct":   round(chg_pct,                            4),
        "change_abs":   round(pm_data.get("change_abs",       0), 2),
        "pm_volume":    pm_data.get("pm_volume", 0),
        "direction":    "up" if chg_pct > 0 else "down",
        "reason":       reason,
        "sector":       result.get("sector",   "—") if result else "—",
        "upside":       result.get("upside",    0)  if result else 0,
        "setups":       result.get("setups",   [])  if result else [],
        "pre_signals":  result.get("pre_signals", {}) if result else {},
        "timestamp":    datetime.datetime.utcnow().isoformat() + "Z",
    }


def _alert_key(ticker: str, direction: str, tier: int) -> str:
    dt = _now_et().strftime("%Y%m%d%H")
    return f"{ticker}:{direction}:{tier}:{dt}"


def _build_reason(
    ticker:     str,
    pd_data:    Dict,
    result:     Optional[Dict],
    tier:       int,
    watchlist:  Set[str],
    recent_set: Set[str],
) -> str:
    reasons = []
    chg_pct = pd_data.get("change_pct", 0) * 100

    if ticker in watchlist:
        reasons.append("on your watchlist")
    if ticker in recent_set:
        reasons.append("fired an alert last scan")

    if result:
        ps    = result.get("pre_signals", {})
        setups = result.get("setups", [])
        micro  = result.get("microstructure", {}).get("microstructure_score", 0)
        n_sig  = ps.get("signal_count", 0)
        if n_sig >= 2:
            top = ps.get("top_signal", "")
            reasons.append(f"{n_sig} pre-signals" +
                           (f" (lead: {top.replace('_',' ')})" if top else ""))
        if "short_squeeze" in setups:
            reasons.append("short squeeze setup")
        if micro >= 0.60:
            reasons.append(f"microstructure {micro:.0%}")

    if not reasons:
        reasons.append(f"unusual pre-market move ({chg_pct:+.1f}%)")

    return " · ".join(reasons)


# ── Notification ──────────────────────────────────────────────────────────────

def _notify_telegram(alert: Dict):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return
    try:
        import requests as _req
        chg   = alert["change_pct"] * 100
        arrow = "🟢" if alert["direction"] == "up" else "🔴"
        msg   = (
            f"{arrow} <b>{alert['ticker']}</b> — Pre-Market {alert['tier_label']}\n"
            f"Price: <b>${alert['pm_price']:.2f}</b> "
            f"({'+' if chg>0 else ''}{chg:.2f}% from ${alert['prev_close']:.2f})\n"
            f"Reason: {alert['reason']}\n"
            f"Sector: {alert['sector']}"
        )
        if alert["setups"]:
            msg += f"\nSetups: {', '.join(alert['setups'])}"
        _req.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=8,
        )
    except Exception as e:
        logger.warning("Telegram PM notification failed: %s", e)


# ── Core scanner ──────────────────────────────────────────────────────────────

def run_premarket_scan(
    watchlist:     Optional[List[str]] = None,
    recent_alerts: Optional[List[str]] = None,
    all_results:   Optional[List[Dict]] = None,
) -> Dict:
    """
    Main pre-market scan. All parameters are OPTIONAL.
    Scanner runs the full universe even with no inputs.
    """
    global _hot_alerts, _discovery_alerts, _last_pm_scan, _pm_scan_count, _fired_keys

    t0 = time.monotonic()
    _pm_scan_count += 1
    logger.info("Pre-market scan #%d starting", _pm_scan_count)

    # ── Build universe independently ──────────────────────────────────────────
    pm_universe  = _get_pm_universe()
    result_map   = {r["ticker"]: r for r in (all_results or [])}

    # ── Tier 1: Watchlist + Recent Alerts + top universe ─────────────────────
    tier1_tickers = _get_tier1_tickers(
        watchlist     = watchlist or [],
        recent_alerts = recent_alerts or [],
        pm_universe   = pm_universe,
    )
    tier1_set = set(tier1_tickers)

    # ── Tier 2: Remaining universe with qualifying signals ────────────────────
    tier2_tickers = [
        t for t in pm_universe
        if t not in tier1_set
        and _tier2_qualifies(result_map.get(t))
    ][:80]

    logger.info("PM scan: %d tier1 tickers, %d tier2 tickers",
                len(tier1_tickers), len(tier2_tickers))

    all_tickers = list(dict.fromkeys(tier1_tickers + tier2_tickers))
    if not all_tickers:
        logger.warning("PM scan: universe empty")
        return {"hot": [], "discovery": [], "meta": premarket_status()}

    pm_data = _fetch_pm_data(all_tickers)
    logger.info("Got PM data for %d/%d tickers", len(pm_data), len(all_tickers))

    hot_alerts:       List[Dict] = []
    discovery_alerts: List[Dict] = []
    notified:         List[Dict] = []
    recent_set = set(t.upper() for t in (recent_alerts or []))
    wl_set     = set(t.upper() for t in (watchlist or []))

    # ── Process Tier 1 ────────────────────────────────────────────────────────
    for ticker in tier1_tickers:
        pd = pm_data.get(ticker)
        if not pd:
            continue
        chg = abs(pd["change_pct"])
        if chg < _TIER1_MOVE_PCT:
            continue
        result = result_map.get(ticker)
        reason = _build_reason(ticker, pd, result, 1, wl_set, recent_set)
        alert  = _build_alert(ticker, pd, 1, result, reason)
        key    = _alert_key(ticker, alert["direction"], 1)
        if key not in _fired_keys:
            hot_alerts.append(alert)
            _fired_keys.add(key)
            notified.append(alert)

    # ── Process Tier 2 ────────────────────────────────────────────────────────
    for ticker in tier2_tickers:
        pd = pm_data.get(ticker)
        if not pd:
            continue
        chg = abs(pd["change_pct"])
        if chg < _TIER2_MOVE_PCT:
            continue
        result = result_map.get(ticker)
        reason = _build_reason(ticker, pd, result, 2, wl_set, recent_set)
        alert  = _build_alert(ticker, pd, 2, result, reason)
        key    = _alert_key(ticker, alert["direction"], 2)
        if key not in _fired_keys:
            discovery_alerts.append(alert)
            _fired_keys.add(key)
            notified.append(alert)

    hot_alerts.sort(      key=lambda x: abs(x["change_pct"]), reverse=True)
    discovery_alerts.sort(key=lambda x: abs(x["change_pct"]), reverse=True)

    for alert in notified:
        try:
            _notify_telegram(alert)
        except Exception as e:
            logger.warning("Notification failed %s: %s", alert["ticker"], e)

    with _pm_lock:
        _hot_alerts       = hot_alerts
        _discovery_alerts = discovery_alerts
        _last_pm_scan     = datetime.datetime.utcnow().isoformat() + "Z"

    # Cache result
    result_dict = {
        "hot":       hot_alerts,
        "discovery": discovery_alerts,
        "meta":      premarket_status(),
        "duration_s": round(time.monotonic() - t0, 1),
    }
    _cache.cache_premarket(result_dict)

    logger.info("PM scan done: %d hot, %d discovery, %.1fs",
                len(hot_alerts), len(discovery_alerts),
                time.monotonic() - t0)
    return result_dict


def get_premarket_results() -> Dict:
    with _pm_lock:
        cached = _cache.get_premarket_cached()
        if cached:
            return {**cached, "last_scan": _last_pm_scan}
        return {
            "hot":       list(_hot_alerts),
            "discovery": list(_discovery_alerts),
            "meta":      premarket_status(),
            "last_scan": _last_pm_scan,
        }


# ── Background thread ─────────────────────────────────────────────────────────

def start_premarket_thread(
    get_watchlist_fn:      Callable[[], List[str]],
    get_recent_alerts_fn:  Callable[[], List[str]],
    get_all_results_fn:    Callable[[], List[Dict]],
):
    """
    Start background pre-market scanner.
    All getter functions are called each scan to get latest state.
    Scanner runs even if getters return empty lists.
    """
    global _pm_thread, _pm_running

    if _pm_running:
        logger.info("Pre-market thread already running")
        return

    _pm_running = True

    def _loop():
        logger.info("Pre-market scanner thread started")
        while _pm_running:
            try:
                if is_premarket():
                    run_premarket_scan(
                        watchlist     = get_watchlist_fn(),
                        recent_alerts = get_recent_alerts_fn(),
                        all_results   = get_all_results_fn(),
                    )
                else:
                    logger.debug("Pre-market scanner idle (outside 4–9:30 AM ET)")
            except Exception as e:
                logger.error("Pre-market scan error: %s", e, exc_info=True)

            # Adaptive sleep: check every 60s near start time, else 5min
            now_et = _now_et()
            start_dt = now_et.replace(
                hour=_PRE_MARKET_START_ET.hour,
                minute=_PRE_MARKET_START_ET.minute,
                second=0, microsecond=0)
            secs_to_start = (start_dt - now_et).total_seconds()
            sleep_s = 60 if 0 < secs_to_start < 600 else _SCAN_INTERVAL_S
            time.sleep(sleep_s)

    _pm_thread = threading.Thread(target=_loop, daemon=True, name="pre_market")
    _pm_thread.start()
    logger.info("Pre-market scanner thread launched (5-min cadence, 4–9:30 AM ET)")


def stop_premarket_thread():
    global _pm_running
    _pm_running = False
