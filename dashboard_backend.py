# ─────────────────────────────────────────────
#  dashboard_backend.py  –  Data aggregation
#
#  Merges yfinance scan results with Webull
#  real-time quotes. Exposes get_dashboard_state()
#  for the Flask API to call on every request.
# ─────────────────────────────────────────────
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config import WEBULL_ENABLED

logger = logging.getLogger("dashboard_backend")


# ── Real-time quote enrichment ────────────────────────────────────────────────

def _enrich_with_rt(result: Dict, rt_quotes: Dict[str, Dict]) -> Dict:
    """
    Overlay Webull real-time quote onto a yfinance scan result.
    Updates: last_price, pre/post market data, volume, bid/ask.
    yfinance scores remain untouched.
    """
    ticker = result.get("ticker", "")
    qt     = rt_quotes.get(ticker)

    if not qt or not qt.get("available"):
        return result

    enriched = dict(result)

    # Only update price if Webull has a valid last
    if qt.get("last") and qt["last"] > 0:
        enriched["last_price"]      = round(qt["last"], 2)
        enriched["rt_bid"]          = qt.get("bid")
        enriched["rt_ask"]          = qt.get("ask")
        enriched["rt_volume"]       = qt.get("volume")
        enriched["rt_change_pct"]   = qt.get("change_pct")
        enriched["rt_source"]       = "webull"

    # Pre/post market
    if qt.get("pre_market_price") and qt["pre_market_price"] > 0:
        enriched["pre_market_price"]   = qt["pre_market_price"]
        enriched["pre_market_chg_pct"] = qt.get("pre_market_chg_pct", 0)
    if qt.get("post_market_price") and qt["post_market_price"] > 0:
        enriched["post_market_price"]   = qt["post_market_price"]
        enriched["post_market_chg_pct"] = qt.get("post_market_chg_pct", 0)

    return enriched


def _enrich_list(results: List[Dict], rt_quotes: Dict[str, Dict]) -> List[Dict]:
    return [_enrich_with_rt(r, rt_quotes) for r in results]


# ── Section builders ──────────────────────────────────────────────────────────

def _build_top_alerts(
    alerts:    List[Dict],
    rt_quotes: Dict[str, Dict],
) -> List[Dict]:
    return _enrich_list(alerts, rt_quotes)


def _build_under10(
    under10:   List[Dict],
    rt_quotes: Dict[str, Dict],
) -> List[Dict]:
    return _enrich_list(under10, rt_quotes)


def _build_scanners(
    scanners:  Dict[str, List[Dict]],
    rt_quotes: Dict[str, Dict],
) -> Dict[str, List[Dict]]:
    return {
        key: _enrich_list(items, rt_quotes)
        for key, items in scanners.items()
    }


def _build_positions_section(rt_quotes: Dict[str, Dict]) -> Dict:
    """Load positions and attach live P&L from Webull quotes."""
    try:
        from positions import get_all_positions, enrich_with_pnl, get_portfolio_summary
        all_pos   = get_all_positions()
        price_map = {t: q["last"] for t, q in rt_quotes.items() if q.get("last", 0) > 0}
        enriched  = enrich_with_pnl(all_pos, price_map)
        return {
            "positions": enriched,
            "summary":   get_portfolio_summary(enriched),
        }
    except Exception as e:
        logger.debug("Positions section error: %s", e)
        return {"positions": [], "summary": {}}


# ── Master aggregator ─────────────────────────────────────────────────────────

def get_dashboard_state(
    last_results:  List[Dict],
    last_alerts:   List[Dict],
    last_under20:  List[Dict],
    last_regime:   Dict,
    last_scanners: Dict,
    scan_stats:    Dict,
    index_data:    Dict,
) -> Dict:
    """
    Merge yfinance scan state with Webull real-time quotes.
    Called by app.py's /api/results endpoint.
    Returns the complete dashboard payload.
    """
    # Get all cached Webull real-time quotes
    rt_quotes: Dict[str, Dict] = {}
    if WEBULL_ENABLED:
        try:
            from scheduler import get_rt_cache
            rt_quotes = get_rt_cache().get("quotes", {})
        except ImportError:
            pass

    # Enrich each section with live quotes
    top_alerts = _build_top_alerts(last_alerts[:20], rt_quotes)
    under10    = _build_under10(last_under20[:20],   rt_quotes)
    all_top    = _enrich_list(last_results,          rt_quotes)
    scanners   = _build_scanners(last_scanners,      rt_quotes)
    positions  = _build_positions_section(rt_quotes)

    # Signal summary
    signal_summary: Dict = {}
    try:
        from signal_report import get_mini_summary
        signal_summary = get_mini_summary()
    except Exception:
        pass

    # Pre-market alerts
    premarket: Dict = {}
    try:
        from pre_market import get_premarket_results
        premarket = get_premarket_results()
    except Exception:
        pass

    return {
        # Sections
        "top_alerts":      top_alerts,
        "alerts":          top_alerts,        # alias
        "under10":         under10,
        "under20":         under10,           # alias
        "scanners":        scanners,
        "top":             all_top,
        # Meta
        "regime":          last_regime,
        "scan_stats":      scan_stats,
        "index_snapshot":  index_data,
        "total":           len(last_results),
        # Real-time enrichment info
        "rt_enabled":      WEBULL_ENABLED and bool(rt_quotes),
        "rt_quote_count":  len(rt_quotes),
        # Features
        "positions":       positions,
        "signal_summary":  signal_summary,
        "premarket":       premarket,
    }


# ── Setup card ────────────────────────────────────────────────────────────────

def get_setup_card(ticker: str, result: Optional[Dict], rt_quotes: Dict) -> Dict:
    """
    Build a full trade setup card merging yfinance historical
    data with Webull real-time price.
    """
    if not result:
        return {"error": f"{ticker} not in last scan results"}

    # Use Webull live price if available, otherwise fall back to yfinance
    rt = rt_quotes.get(ticker, {})
    live_price = rt.get("last", 0) if rt.get("available") else 0

    try:
        from data_yfinance import get_daily_ohlcv
        from trade_setup import compute_trade_setup
        candles    = get_daily_ohlcv(ticker, lookback_days=120)
        stock_data = {"prices": [c["close"] for c in candles], "atr_pct": 0.02}
        if candles:
            from data_yfinance import compute_atr14
            atr14 = compute_atr14(candles)
            price = live_price or result.get("last_price", 1)
            stock_data["atr_pct"] = atr14 / price if price else 0.02
            stock_data["candles"] = candles

        plan = compute_trade_setup(result, stock_data)

        # Overlay live price
        if live_price > 0:
            plan["last_price"]  = round(live_price, 2)
            plan["rt_price"]    = round(live_price, 2)
            plan["price_source"] = "webull_rt"
        else:
            plan["price_source"] = "yfinance"

        return plan
    except Exception as e:
        logger.warning("get_setup_card %s: %s", ticker, e)
        return {"error": str(e)}


def log_trade_from_setup(ticker: str, entry_price: float, setup: Dict) -> Dict:
    """
    Create a position from a Setup Card confirmation.
    Called when user clicks Trade Now and confirms.
    """
    try:
        from positions import add_position
        pos = add_position({
            "ticker":     ticker,
            "entry":      entry_price,
            "stop":       setup.get("stop", 0),
            "target1":    setup.get("target1", 0),
            "target2":    setup.get("target2", 0),
            "shares":     setup.get("suggested_shares", 0),
            "setup_type": setup.get("primary_setup", ""),
            "notes":      f"Setup card: {setup.get('entry_type','')}, "
                          f"confidence {setup.get('confidence',0)}/5",
        })
        return {"status": "logged", "position": pos}
    except Exception as e:
        return {"status": "error", "message": str(e)}
