# ─────────────────────────────────────────────
#  signal_report.py  –  Alert Quality Scorecard
#
#  Reads alert_history.jsonl and grades past
#  alerts by checking what price did after.
#
#  Lookback windows: 1-day, 3-day, 5-day
#  Win threshold: +2% move in predicted direction
#
#  Exposed via /report page and /api/report.
# ─────────────────────────────────────────────
from __future__ import annotations

import datetime
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

from config import HISTORY_CONFIG, REQUEST_DELAY_SECONDS

logger = logging.getLogger("signal_report")

_ALERT_LOG   = HISTORY_CONFIG.get("alert_log_file", "alert_history.jsonl")
_WIN_PCT     = 0.02    # 2% minimum to count as a win
_CACHE_FILE  = "report_cache.json"
_CACHE_TTL   = 3600    # rebuild cache at most once per hour


# ── Price fetcher ─────────────────────────────────────────────────────────────

def _fetch_price_on_date(ticker: str, target_date: datetime.date) -> Optional[float]:
    """
    Fetch closing price for ticker on or near target_date.
    Returns None if unavailable.
    """
    if not YF_AVAILABLE:
        return None
    try:
        time.sleep(REQUEST_DELAY_SECONDS)
        end   = target_date + datetime.timedelta(days=3)
        hist  = yf.Ticker(ticker).history(
            start=target_date.isoformat(),
            end=end.isoformat(),
        )
        if hist.empty:
            return None
        return float(hist["Close"].iloc[0])
    except Exception as e:
        logger.debug("Price fetch failed %s %s: %s", ticker, target_date, e)
        return None


def _fetch_current_price(ticker: str) -> Optional[float]:
    """Fetch current price for open alerts."""
    if not YF_AVAILABLE:
        return None
    try:
        time.sleep(REQUEST_DELAY_SECONDS)
        info = yf.Ticker(ticker).info or {}
        return float(info.get("currentPrice") or info.get("regularMarketPrice") or 0) or None
    except Exception:
        return None


# ── Cache ─────────────────────────────────────────────────────────────────────

def _load_cache() -> Optional[Dict]:
    if not os.path.exists(_CACHE_FILE):
        return None
    try:
        with open(_CACHE_FILE) as f:
            data = json.load(f)
        if time.time() - data.get("built_at", 0) < _CACHE_TTL:
            return data
    except Exception:
        pass
    return None


def _save_cache(data: Dict):
    try:
        data["built_at"] = time.time()
        with open(_CACHE_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning("Could not save report cache: %s", e)


# ── Alert loader ──────────────────────────────────────────────────────────────

def _load_alerts(limit: int = 200) -> List[Dict]:
    if not os.path.exists(_ALERT_LOG):
        return []
    try:
        records = []
        with open(_ALERT_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records[-limit:]
    except Exception as e:
        logger.warning("Could not read alert log: %s", e)
        return []


# ── Core grader ───────────────────────────────────────────────────────────────

def _grade_alert(alert: Dict, today: datetime.date) -> Dict:
    """
    Grade a single alert against 1/3/5-day price performance.
    Returns the alert enriched with performance data.
    """
    ticker     = alert.get("ticker", "")
    ts_str     = alert.get("timestamp", "")
    alert_price= float(alert.get("price") or 0)

    # Parse alert date
    try:
        alert_dt = datetime.datetime.fromisoformat(ts_str.replace("Z",""))
        alert_date = alert_dt.date()
    except (ValueError, TypeError):
        return {**alert, "grade": "ungraded", "error": "bad timestamp"}

    days_since = (today - alert_date).days

    result = {**alert, "alert_date": alert_date.isoformat(), "days_since": days_since}

    # Fetch prices at each lookback window
    for window, label in [(1, "d1"), (3, "d3"), (5, "d5")]:
        if days_since < window:
            result[f"{label}_price"]   = None
            result[f"{label}_pct"]     = None
            result[f"{label}_win"]     = None
            continue

        target_date = alert_date + datetime.timedelta(days=window)
        price       = _fetch_price_on_date(ticker, target_date)

        if price and alert_price > 0:
            pct  = (price - alert_price) / alert_price
            win  = pct >= _WIN_PCT
            result[f"{label}_price"] = round(price, 2)
            result[f"{label}_pct"]   = round(pct * 100, 2)
            result[f"{label}_win"]   = win
        else:
            result[f"{label}_price"] = None
            result[f"{label}_pct"]   = None
            result[f"{label}_win"]   = None

    # Overall grade
    wins = [result[f"{l}_win"] for l in ("d1","d3","d5") if result[f"{l}_win"] is not None]
    if not wins:
        result["grade"] = "pending"
    elif sum(wins) >= 2:
        result["grade"] = "strong_win"
    elif wins[-1]:   # latest window is a win
        result["grade"] = "win"
    elif sum(wins) == 0:
        result["grade"] = "loss"
    else:
        result["grade"] = "mixed"

    return result


# ── Aggregator ────────────────────────────────────────────────────────────────

def _aggregate(graded: List[Dict]) -> Dict:
    """Compute summary stats from graded alerts."""
    scored = [g for g in graded if g.get("grade") not in ("pending","ungraded","error")]

    def win_rate(window: str) -> Optional[float]:
        vals = [g for g in scored if g.get(f"{window}_win") is not None]
        if not vals: return None
        return round(sum(1 for g in vals if g[f"{window}_win"]) / len(vals) * 100, 1)

    def avg_pct(window: str) -> Optional[float]:
        vals = [g[f"{window}_pct"] for g in scored if g.get(f"{window}_pct") is not None]
        if not vals: return None
        return round(float(np.mean(vals)), 2)

    # By setup type
    setup_stats: Dict[str, Dict] = {}
    for g in scored:
        for setup in (g.get("setups") or []):
            if setup not in setup_stats:
                setup_stats[setup] = {"total": 0, "wins": 0}
            setup_stats[setup]["total"] += 1
            if g.get("d3_win"):
                setup_stats[setup]["wins"] += 1
    for s in setup_stats:
        t = setup_stats[s]
        t["win_rate"] = round(t["wins"]/t["total"]*100, 1) if t["total"] else 0

    # By sector
    sector_stats: Dict[str, Dict] = {}
    for g in scored:
        sector = g.get("sector", "Unknown") or "Unknown"
        if sector not in sector_stats:
            sector_stats[sector] = {"total": 0, "wins": 0}
        sector_stats[sector]["total"] += 1
        if g.get("d3_win"):
            sector_stats[sector]["wins"] += 1
    for s in sector_stats:
        t = sector_stats[s]
        t["win_rate"] = round(t["wins"]/t["total"]*100, 1) if t["total"] else 0

    # Recent 20 alerts mini summary
    recent = sorted(scored, key=lambda x: x.get("alert_date",""), reverse=True)[:20]
    recent_wins = sum(1 for g in recent if g.get("d3_win"))
    recent_wr   = round(recent_wins / len(recent) * 100, 1) if recent else None

    return {
        "total_alerts":   len(graded),
        "graded_alerts":  len(scored),
        "pending_alerts": len(graded) - len(scored),
        "win_rate_d1":    win_rate("d1"),
        "win_rate_d3":    win_rate("d3"),
        "win_rate_d5":    win_rate("d5"),
        "avg_pct_d1":     avg_pct("d1"),
        "avg_pct_d3":     avg_pct("d3"),
        "avg_pct_d5":     avg_pct("d5"),
        "setup_stats":    setup_stats,
        "sector_stats":   sector_stats,
        "recent_20_wr":   recent_wr,
        "recent_20_wins": recent_wins if recent else 0,
        "recent_20_total":len(recent),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def build_report(force: bool = False) -> Dict:
    """
    Build the full signal quality report.
    Uses cache unless force=True or cache is stale (>1 hour).
    """
    if not force:
        cached = _load_cache()
        if cached:
            logger.info("Returning cached signal report")
            return cached

    logger.info("Building signal report from %s", _ALERT_LOG)
    today   = datetime.date.today()
    alerts  = _load_alerts(limit=200)

    if not alerts:
        result = {
            "graded":    [],
            "summary":   {"total_alerts": 0, "graded_alerts": 0},
            "generated": datetime.datetime.utcnow().isoformat() + "Z",
        }
        _save_cache(result)
        return result

    # Grade each alert (fetches prices — can be slow)
    graded = []
    for alert in alerts:
        try:
            graded.append(_grade_alert(alert, today))
        except Exception as e:
            logger.warning("Grade failed for %s: %s", alert.get("ticker"), e)

    summary = _aggregate(graded)

    result = {
        "graded":    graded,
        "summary":   summary,
        "generated": datetime.datetime.utcnow().isoformat() + "Z",
        "win_threshold_pct": _WIN_PCT * 100,
    }
    _save_cache(result)
    logger.info("Report built: %d alerts, d3 win rate %.1f%%",
                len(graded), summary.get("win_rate_d3") or 0)
    return result


def get_mini_summary() -> Dict:
    """
    Fast summary for the dashboard widget.
    Uses cache — does not trigger a full rebuild.
    """
    cached = _load_cache()
    if cached:
        s = cached.get("summary", {})
        return {
            "recent_20_wr":    s.get("recent_20_wr"),
            "avg_pct_d1":      s.get("avg_pct_d1"),
            "avg_pct_d3":      s.get("avg_pct_d3"),
            "total_alerts":    s.get("total_alerts", 0),
            "graded_alerts":   s.get("graded_alerts", 0),
            "generated":       cached.get("generated"),
        }
    return {"recent_20_wr": None, "avg_pct_d3": None, "total_alerts": 0}
