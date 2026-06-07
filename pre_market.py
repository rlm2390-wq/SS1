# ─────────────────────────────────────────────
#  pre_market.py  –  Pre-Market Alerter
#
#  Runs on a 5-minute cadence, 4:00–9:30 AM ET.
#  Two-tier alert system:
#
#  Tier 1 — Watchlist + Recent Alerts
#    Fire immediately on any meaningful move.
#    Threshold: >1.5% pre-market move.
#    Shown in "🔥 Hot Pre-Market" section.
#
#  Tier 2 — Full Universe
#    Fire only if 2+ pre-signals OR explosive
#    setup OR volume anomaly OR microstructure
#    pressure. Threshold: >1.0% move.
#    Shown in "🌅 Pre-Market Discovery" section.
#
#  Data source: yfinance preMarketPrice /
#  preMarketChange (already fetched per ticker).
#  No extra API calls — uses snapshot data.
# ─────────────────────────────────────────────
from __future__ import annotations

import datetime
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set

import numpy as np

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

from config import (REQUEST_DELAY_SECONDS, REQUEST_MAX_RETRIES,
                    REQUEST_RETRY_BACKOFF, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

logger = logging.getLogger("pre_market")

# ── Timing ────────────────────────────────────────────────────────────────────
_PRE_MARKET_START_ET = datetime.time(4,  0)   # 4:00 AM ET
_PRE_MARKET_END_ET   = datetime.time(9, 30)   # 9:30 AM ET
_SCAN_INTERVAL_S     = 300                    # 5 minutes

# ── Thresholds ────────────────────────────────────────────────────────────────
_TIER1_MOVE_PCT      = 0.015   # 1.5% — watchlist/recent alerts
_TIER2_MOVE_PCT      = 0.010   # 1.0% — universe discovery
_TIER2_MIN_SIGNALS   = 2       # minimum pre-signals for tier 2

# ── Module state ──────────────────────────────────────────────────────────────
_pm_lock             = threading.Lock()
_pm_thread: Optional[threading.Thread] = None
_pm_running          = False

# Alert results — read by app.py for the API
_hot_alerts:       List[Dict] = []   # Tier 1
_discovery_alerts: List[Dict] = []   # Tier 2
_last_pm_scan:     Optional[str] = None
_pm_scan_count     = 0
_fired_keys:       Set[str] = set()  # prevent duplicate notifications


# ── ET timezone helper ────────────────────────────────────────────────────────

def _now_et() -> datetime.datetime:
    """Return current time in US/Eastern (handles DST manually via UTC offset)."""
    utc_now = datetime.datetime.utcnow()
    # EST = UTC-5, EDT = UTC-4 (rough — good enough for market hours)
    # March second Sunday → November first Sunday = EDT (-4)
    month = utc_now.month
    is_edt = 3 <= month <= 11
    offset = -4 if is_edt else -5
    return utc_now + datetime.timedelta(hours=offset)


def is_premarket() -> bool:
    """Return True if current ET time is within pre-market window."""
    now_et = _now_et().time()
    return _PRE_MARKET_START_ET <= now_et < _PRE_MARKET_END_ET


def premarket_status() -> Dict:
    """Return current pre-market status info."""
    now_et = _now_et()
    active = is_premarket()
    start  = now_et.replace(
        hour=_PRE_MARKET_START_ET.hour,
        minute=_PRE_MARKET_START_ET.minute, second=0, microsecond=0)
    end    = now_et.replace(
        hour=_PRE_MARKET_END_ET.hour,
        minute=_PRE_MARKET_END_ET.minute, second=0, microsecond=0)

    if active:
        mins_left = int((end - now_et).total_seconds() / 60)
        status_str = f"Active — {mins_left}m until market open"
    else:
        if now_et.time() < _PRE_MARKET_START_ET:
            mins_to   = int((start - now_et).total_seconds() / 60)
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


# ── yfinance pre-market fetch ─────────────────────────────────────────────────

def _fetch_premarket_batch(tickers: List[str]) -> Dict[str, Dict]:
    """
    Fetch pre-market price and change for a list of tickers.
    Returns {ticker: {price, prev_close, change_pct, volume}} dict.
    Sequential with delay to avoid rate limits.
    """
    results = {}
    if not YF_AVAILABLE:
        return results

    for ticker in tickers:
        delay = REQUEST_DELAY_SECONDS
        for attempt in range(REQUEST_MAX_RETRIES):
            try:
                time.sleep(delay)
                t    = yf.Ticker(ticker)
                info = t.info or {}

                prev_close  = float(info.get("previousClose")       or
                                    info.get("regularMarketPreviousClose") or 0)
                pm_price    = float(info.get("preMarketPrice")       or 0)
                pm_chg      = float(info.get("preMarketChange")      or 0)
                pm_chg_pct  = float(info.get("preMarketChangePercent") or
                                    (pm_chg / prev_close if prev_close else 0))
                pm_vol      = int(info.get("preMarketVolume")        or 0)
                reg_price   = float(info.get("regularMarketPrice")   or
                                    info.get("currentPrice")         or 0)

                if pm_price > 0:
                    results[ticker] = {
                        "price":      pm_price,
                        "prev_close": prev_close,
                        "change_pct": pm_chg_pct,
                        "change_abs": pm_chg,
                        "volume":     pm_vol,
                        "reg_price":  reg_price,
                    }
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


# ── Alert builders ────────────────────────────────────────────────────────────

def _build_alert(ticker: str, pm_data: Dict, tier: int,
                 result: Optional[Dict], reason: str) -> Dict:
    """Build a pre-market alert dict."""
    chg_pct = pm_data.get("change_pct", 0)
    return {
        "ticker":       ticker,
        "tier":         tier,
        "tier_label":   "🔥 Hot" if tier == 1 else "🌅 Discovery",
        "pm_price":     round(pm_data.get("price",      0), 2),
        "prev_close":   round(pm_data.get("prev_close", 0), 2),
        "change_pct":   round(chg_pct,                     4),
        "change_abs":   round(pm_data.get("change_abs", 0), 2),
        "pm_volume":    pm_data.get("volume", 0),
        "direction":    "up" if chg_pct > 0 else "down",
        "reason":       reason,
        "sector":       result.get("sector", "—") if result else "—",
        "upside":       result.get("upside", 0)   if result else 0,
        "setups":       result.get("setups", [])  if result else [],
        "pre_signals":  result.get("pre_signals", {}) if result else {},
        "timestamp":    datetime.datetime.utcnow().isoformat() + "Z",
    }


def _alert_key(ticker: str, direction: str, tier: int) -> str:
    """Dedup key — one alert per ticker per direction per tier per session."""
    dt = _now_et().strftime("%Y%m%d%H")   # expires hourly
    return f"{ticker}:{direction}:{tier}:{dt}"


# ── Notification ──────────────────────────────────────────────────────────────

def _notify_telegram(alert: Dict):
    """Fire Telegram notification for a pre-market alert."""
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return
    try:
        import requests
        chg    = alert["change_pct"] * 100
        arrow  = "🟢" if alert["direction"] == "up" else "🔴"
        tier   = alert["tier_label"]
        msg    = (
            f"{arrow} <b>{alert['ticker']}</b> — Pre-Market {tier}\n"
            f"Price: <b>${alert['pm_price']:.2f}</b> "
            f"({'+' if chg>0 else ''}{chg:.2f}% from ${alert['prev_close']:.2f})\n"
            f"Reason: {alert['reason']}\n"
            f"Sector: {alert['sector']}"
        )
        if alert["setups"]:
            msg += f"\nSetups: {', '.join(alert['setups'])}"

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       msg,
            "parse_mode": "HTML",
        }, timeout=8)
        logger.info("Telegram PM alert sent: %s %+.1f%%", alert["ticker"], chg)
    except Exception as e:
        logger.warning("Telegram PM notification failed: %s", e)


# ── Core scanner ─────────────────────────────────────────────────────────────

def run_premarket_scan(
    watchlist:     List[str],
    recent_alerts: List[str],
    all_results:   List[Dict],
) -> Dict:
    """
    Main pre-market scan. Called every 5 minutes while is_premarket() is True.

    Args:
        watchlist:     server-side watchlist tickers
        recent_alerts: tickers that fired in the last full scan
        all_results:   full results list from last main scan (for signal data)

    Returns dict with hot_alerts, discovery_alerts, scan_meta.
    """
    global _hot_alerts, _discovery_alerts, _last_pm_scan, _pm_scan_count, _fired_keys

    t0 = time.monotonic()
    _pm_scan_count += 1
    logger.info("Pre-market scan #%d starting", _pm_scan_count)

    # Build result lookup
    result_map: Dict[str, Dict] = {r["ticker"]: r for r in (all_results or [])}

    # ── Tier 1: Watchlist + Recent Alerts ────────────────────────────────────
    tier1_tickers = list(dict.fromkeys(
        [t.upper() for t in (watchlist or [])] +
        [t.upper() for t in (recent_alerts or [])]
    ))

    # ── Tier 2: Universe (excluding tier 1) ──────────────────────────────────
    tier1_set     = set(tier1_tickers)
    tier2_tickers = [
        r["ticker"] for r in (all_results or [])
        if r["ticker"] not in tier1_set
        and _tier2_qualifies(r)
    ][:80]   # cap to avoid flooding yfinance

    logger.info("PM scan: %d tier1 tickers, %d tier2 tickers",
                len(tier1_tickers), len(tier2_tickers))

    # Fetch pre-market data
    all_tickers = tier1_tickers + tier2_tickers
    if not all_tickers:
        logger.info("No tickers to scan pre-market")
        return {"hot": [], "discovery": [], "meta": premarket_status()}

    pm_data = _fetch_premarket_batch(all_tickers)
    logger.info("Got PM data for %d/%d tickers", len(pm_data), len(all_tickers))

    hot_alerts:       List[Dict] = []
    discovery_alerts: List[Dict] = []
    notified         = []

    # ── Process Tier 1 ────────────────────────────────────────────────────────
    for ticker in tier1_tickers:
        pd = pm_data.get(ticker)
        if not pd:
            continue

        chg = abs(pd["change_pct"])
        if chg < _TIER1_MOVE_PCT:
            continue

        result = result_map.get(ticker)
        reason = _build_reason(ticker, pd, result, tier=1,
                               watchlist=tier1_set,
                               recent_set=set(recent_alerts or []))
        alert  = _build_alert(ticker, pd, tier=1, result=result, reason=reason)
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
        reason = _build_reason(ticker, pd, result, tier=2,
                               watchlist=tier1_set,
                               recent_set=set(recent_alerts or []))
        alert  = _build_alert(ticker, pd, tier=2, result=result, reason=reason)
        key    = _alert_key(ticker, alert["direction"], 2)

        if key not in _fired_keys:
            discovery_alerts.append(alert)
            _fired_keys.add(key)
            notified.append(alert)

    # Sort by move size descending
    hot_alerts.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
    discovery_alerts.sort(key=lambda x: abs(x["change_pct"]), reverse=True)

    # Fire notifications for new alerts
    for alert in notified:
        try:
            _notify_telegram(alert)
        except Exception as e:
            logger.warning("Notification failed for %s: %s", alert["ticker"], e)

    # Update module state
    with _pm_lock:
        _hot_alerts       = hot_alerts
        _discovery_alerts = discovery_alerts
        _last_pm_scan     = datetime.datetime.utcnow().isoformat() + "Z"

    dur = round(time.monotonic() - t0, 1)
    logger.info("PM scan done: %d hot, %d discovery, %.1fs",
                len(hot_alerts), len(discovery_alerts), dur)

    return {
        "hot":       hot_alerts,
        "discovery": discovery_alerts,
        "meta":      premarket_status(),
        "duration_s": dur,
    }


def _tier2_qualifies(result: Dict) -> bool:
    """Return True if a ticker qualifies for tier 2 scanning."""
    ps = result.get("pre_signals", {})
    return (
        ps.get("signal_count", 0) >= _TIER2_MIN_SIGNALS or
        bool(set(result.get("setups", [])) & {"short_squeeze", "volatility_breakout"}) or
        result.get("microstructure", {}).get("microstructure_score", 0) >= 0.68
    )


def _build_reason(ticker: str, pd: Dict, result: Optional[Dict],
                  tier: int, watchlist: Set, recent_set: Set) -> str:
    """Build a plain-English reason string for the alert."""
    reasons = []
    chg_pct = pd.get("change_pct", 0) * 100

    if ticker in watchlist:
        reasons.append("on your watchlist")
    if ticker in recent_set:
        reasons.append("fired an alert last scan")

    if result:
        ps     = result.get("pre_signals", {})
        setups = result.get("setups", [])
        micro  = result.get("microstructure", {}).get("microstructure_score", 0)
        n_sig  = ps.get("signal_count", 0)

        if n_sig >= 2:
            top = ps.get("top_signal", "")
            reasons.append(f"{n_sig} pre-signals active" +
                           (f" (lead: {top.replace('_',' ')})" if top else ""))
        if "short_squeeze" in setups:
            reasons.append("short squeeze setup active")
        if "volatility_breakout" in setups:
            reasons.append("volatility breakout setup active")
        if micro >= 0.68:
            reasons.append(f"strong microstructure ({micro:.0%})")

    if not reasons:
        reasons.append(f"unusual pre-market move ({chg_pct:+.1f}%)")

    return " · ".join(reasons)


# ── Background thread ─────────────────────────────────────────────────────────

def get_premarket_results() -> Dict:
    """Read current pre-market alert state (called by app.py routes)."""
    with _pm_lock:
        return {
            "hot":       list(_hot_alerts),
            "discovery": list(_discovery_alerts),
            "meta":      premarket_status(),
            "last_scan": _last_pm_scan,
        }


def start_premarket_thread(
    get_watchlist_fn,
    get_recent_alerts_fn,
    get_all_results_fn,
):
    """
    Start the background pre-market scanner thread.
    Accepts callables so it always uses current state from app.py.

    Call once at app startup.
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
                        watchlist=get_watchlist_fn(),
                        recent_alerts=get_recent_alerts_fn(),
                        all_results=get_all_results_fn(),
                    )
                else:
                    # Outside pre-market — clear stale alerts, sleep longer
                    with _pm_lock:
                        pass  # keep last results visible until explicitly cleared
                    logger.debug("Pre-market scanner idle (outside hours)")
            except Exception as e:
                logger.error("Pre-market scan error: %s", e, exc_info=True)

            # Sleep 5 minutes, but wake up if pre-market starts soon
            now_et  = _now_et().time()
            minutes = _SCAN_INTERVAL_S / 60

            # If we're within 10 min of pre-market start, check more often
            start_dt = _now_et().replace(
                hour=_PRE_MARKET_START_ET.hour,
                minute=_PRE_MARKET_START_ET.minute,
                second=0, microsecond=0)
            secs_to_start = (start_dt - _now_et()).total_seconds()
            if 0 < secs_to_start < 600:
                sleep_s = 60   # 1-minute check near open
            else:
                sleep_s = _SCAN_INTERVAL_S

            time.sleep(sleep_s)

    _pm_thread = threading.Thread(target=_loop, daemon=True, name="pre_market")
    _pm_thread.start()
    logger.info("Pre-market scanner thread launched (5-min cadence, 4–9:30 AM ET)")


def stop_premarket_thread():
    global _pm_running
    _pm_running = False
