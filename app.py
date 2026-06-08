# ─────────────────────────────────────────────
#  app.py  –  Flask web server
# ─────────────────────────────────────────────
from __future__ import annotations
import sys
import os

# Ensure project root is on sys.path regardless of where gunicorn launches from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datetime
import logging
import threading
import time
from flask import Flask, render_template, jsonify

from config import ALERT_CONFIG, HISTORY_CONFIG
from universe import get_universe
from market_data import get_index_data, get_stock_data, get_sector_stats
from history import HistoryStore, get_global_store
import regime, technical, fundamental, sentiment, structural, risk, setups, scoring, validation
from main import score_ticker, should_alert, is_under20_popper, is_under10_popper
from under10 import filter_and_rank_under10
from scanners import run_all_scanners
from scoring import get_top2_weights, get_current_weights
from pre_market import start_premarket_thread, get_premarket_results, is_premarket, premarket_status
from trade_setup import compute_trade_setup
from positions import (get_all_positions, add_position, close_position,
                        update_position, delete_position,
                        enrich_with_pnl, get_portfolio_summary)
from signal_report import build_report, get_mini_summary

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("app")

app = Flask(__name__, template_folder=".")

# ── Shared scan state ─────────────────────────────────────────────────────────
_scan_lock       = threading.Lock()
_last_results    = []
_last_alerts     = []
_last_under20    = []
_last_under20    = []
_last_pre_signals = []   # tickers with 1+ pre-signals
_last_scanners    = {}   # six discovery scanners
_weight_display   = ""   # e.g. "Tech 34% · Fund 28%"
_watchlist         = []   # server-side watchlist
_last_regime     = {"label": "unknown", "score": 0.0}
_last_scan_time  = None
_is_scanning     = False
_scan_progress   = {"current": 0, "total": 0, "ticker": ""}
_last_scan_stats = {"duration_s": None, "tickers_processed": 0, "tickers_skipped": 0, "scan_mode": ""}
_next_scan_time  = None

_SCAN_INTERVAL       = 600   # 10 minutes (quick scan cadence)
_QUICK_SCAN_INTERVAL = 600   # 10 min — top-100 tickers only
_scan_count          = 0     # incremented each cycle; every 6th = full scan


def schedule_next_scan():
    global _next_scan_time
    _next_scan_time = datetime.datetime.utcnow() + datetime.timedelta(seconds=_SCAN_INTERVAL)
    logger.info("Next scheduled scan at %s UTC", _next_scan_time.strftime("%Y-%m-%dT%H:%M:%S"))
    t = threading.Timer(_SCAN_INTERVAL, run_scan_background)
    t.daemon = True
    t.start()


def run_scan_background():
    global _last_results, _last_alerts, _last_under20, _last_regime, _last_scan_time
    global _is_scanning, _scan_progress, _last_scan_stats, _last_pre_signals, _weight_display, _scan_count, _last_scanners

    with _scan_lock:
        _is_scanning   = True
        _scan_progress = {"current": 0, "total": 0, "ticker": ""}

    scan_start = time.monotonic()

    # Determine scan mode: every 6th cycle is a full scan (60 min cadence),
    # all other cycles are quick scans over the top 100 tickers only.
    _scan_count += 1
    if _scan_count % 6 == 0:
        tickers   = get_universe()
        scan_mode = "full"
        logger.info("Full scan: %d tickers (cycle %d)", len(tickers), _scan_count)
    else:
        tickers   = get_universe()[:100]
        scan_mode = "quick"
        logger.info("Quick scan: 100 tickers (cycle %d)", _scan_count)

    try:
        history_store = get_global_store()

        try:
            index_data = get_index_data()
        except Exception as exc:
            logger.error("Failed to fetch index data: %s", exc, exc_info=True)
            raise

        rl, rs = regime.compute_market_context(index_data, history_store)
        logger.info("Market regime: %s (score=%.3f)", rl, rs)

        with _scan_lock:
            _scan_progress["total"] = len(tickers)

        results, alerts, under20, pre_signals_list = [], [], [], []
        skipped = 0

        for i, ticker in enumerate(tickers):
            if not ticker or not isinstance(ticker, str) or len(ticker) > 10:
                logger.warning("Skipping invalid ticker: %r", ticker)
                skipped += 1
                continue

            with _scan_lock:
                _scan_progress["current"] = i + 1
                _scan_progress["ticker"]  = ticker

            try:
                result = score_ticker(ticker, rl, rs, history_store)
            except Exception as exc:
                logger.warning("score_ticker failed for %s: %s", ticker, exc)
                skipped += 1
                continue

            if result is None:
                skipped += 1
                continue

            results.append(result)

            if should_alert(rs, result["upside"], result["risk"],
                            result["setup_score"],
                            result["upside_change"] if _scan_count > 1 else 999,
                            ALERT_CONFIG):
                alerts.append(result)

            if is_under10_popper(result):
                under20.append(result)

            # Collect pre-signals (1+ signals, tiered by strength)
            ps = result.get("pre_signals", {})
            if ps.get("signal_count", 0) >= 1:
                pre_signals_list.append(result)

        results.sort(key=lambda x: x["upside"], reverse=True)
        alerts.sort(key=lambda x: x["upside"], reverse=True)
        # Apply full Under $10 gates, scoring, and ranking
        under20 = filter_and_rank_under10(under20)

        # Run all six discovery scanners
        scanner_results = run_all_scanners(results)
        pre_signals_list.sort(
            key=lambda x: (
                x.get("pre_signals", {}).get("pre_score", 0) *
                x.get("pre_signals", {}).get("signal_count", 0)
            ),
            reverse=True
        )

        scan_duration = round(time.monotonic() - scan_start, 2)
        logger.info(
            "Scan complete: %d processed, %d skipped, %d alerts, %d under20, %.1fs",
            len(results), skipped, len(alerts), len(under20), scan_duration,
        )

        # Persist history store to disk after every scan
        history_store.save()

        with _scan_lock:
            _last_results    = results
            _last_alerts     = alerts
            _last_under20    = under20
            _last_pre_signals = pre_signals_list[:25]
            _last_scanners    = scanner_results
            _weight_display   = ""
            _last_regime     = {"label": rl, "score": round(rs, 3)}
            _last_scan_time  = datetime.datetime.utcnow().isoformat() + "Z"
            _last_scan_stats = {
                "duration_s":        scan_duration,
                "tickers_processed": len(results),
                "tickers_skipped":   skipped,
                "scan_mode":         scan_mode,
            }

    except Exception as exc:
        logger.error("Scan failed: %s", exc, exc_info=True)
    finally:
        with _scan_lock:
            _is_scanning = False
        schedule_next_scan()


# Run initial scan on startup
threading.Thread(target=run_scan_background, daemon=True).start()

# Start pre-market alerter (runs 4–9:30 AM ET on 5-min cadence)
start_premarket_thread(
    get_watchlist_fn     = lambda: list(_watchlist),
    get_recent_alerts_fn = lambda: [a["ticker"] for a in _last_alerts],
    get_all_results_fn   = lambda: list(_last_results),
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/scan", methods=["POST"])
def trigger_scan():
    with _scan_lock:
        if _is_scanning:
            logger.info("Scan requested but already running — ignoring")
            return jsonify({"status": "already_running"})
    logger.info("Manual scan triggered via /api/scan")
    threading.Thread(target=run_scan_background, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/results")
def api_results():
    with _scan_lock:
        next_scan_iso = _next_scan_time.isoformat() + "Z" if _next_scan_time else None
        return jsonify({
            "regime":         _last_regime,
            "scan_time":      _last_scan_time,
            "next_scan_time": next_scan_iso,
            "is_scanning":    _is_scanning,
            "progress":       dict(_scan_progress),
            "total":          len(_last_results),
            "alerts":         _last_alerts[:20],
            "under10":        _last_under20[:20],
            "under20":        _last_under20[:20],   # kept for backward compat
            "top":            _last_results,
            "pre_signals":    _last_pre_signals,
            "weight_display": _weight_display,
            "scanners":       _last_scanners,
            "premarket":      get_premarket_results(),
            "signal_summary":  get_mini_summary(),
            "scan_stats":     dict(_last_scan_stats),
        })




@app.route("/api/premarket")
def api_premarket():
    """Return current pre-market alert state for both tiers."""
    data = get_premarket_results()
    data["is_premarket"] = is_premarket()
    return jsonify(data)


@app.route("/api/premarket/scan", methods=["POST"])
def trigger_premarket_scan():
    """Manually trigger a pre-market scan (for testing outside market hours)."""
    import threading
    from pre_market import run_premarket_scan
    def _run():
        run_premarket_scan(
            watchlist     = list(_watchlist) if '_watchlist' in dir() else [],
            recent_alerts = [a["ticker"] for a in _last_alerts],
            all_results   = list(_last_results),
        )
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started"})



# ── Trade Setup ───────────────────────────────────────────────────────────────

@app.route("/api/setup/<ticker>")
def api_trade_setup(ticker: str):
    """Compute a full trade setup card for a ticker."""
    from market_data import get_stock_data
    from config import HISTORY_CONFIG
    ticker = ticker.upper()
    with _scan_lock:
        result = next((r for r in _last_results if r["ticker"] == ticker), None)
    if not result:
        return jsonify({"error": "Ticker not in last scan results"}), 404
    try:
        stock_data = get_stock_data(ticker, HISTORY_CONFIG["lookback_days"])
        setup      = compute_trade_setup(result, stock_data)
        return jsonify(setup)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Position Tracker ──────────────────────────────────────────────────────────

@app.route("/api/positions", methods=["GET"])
def api_get_positions():
    positions = get_all_positions()
    with _scan_lock:
        price_map = {r["ticker"]: r.get("last_price", 0) for r in _last_results}
    enriched  = enrich_with_pnl(positions, price_map)
    summary   = get_portfolio_summary(enriched)
    return jsonify({"positions": enriched, "summary": summary})


@app.route("/api/positions", methods=["POST"])
def api_add_position():
    data = request.get_json(silent=True) or {}
    try:
        pos = add_position(data)
        return jsonify({"position": pos, "status": "added"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/positions/<pos_id>", methods=["PUT"])
def api_update_position(pos_id: str):
    data = request.get_json(silent=True) or {}
    pos  = update_position(pos_id, data)
    if pos:
        return jsonify({"position": pos, "status": "updated"})
    return jsonify({"error": "Position not found"}), 404


@app.route("/api/positions/<pos_id>/close", methods=["POST"])
def api_close_position(pos_id: str):
    data       = request.get_json(silent=True) or {}
    exit_price = float(data.get("exit_price", 0))
    exit_reason= data.get("reason", "manual")
    pos        = close_position(pos_id, exit_price, exit_reason)
    if pos:
        return jsonify({"position": pos, "status": "closed"})
    return jsonify({"error": "Position not found or already closed"}), 404


@app.route("/api/positions/<pos_id>", methods=["DELETE"])
def api_delete_position(pos_id: str):
    ok = delete_position(pos_id)
    return jsonify({"status": "deleted" if ok else "not_found"})


# ── Signal Report ─────────────────────────────────────────────────────────────

@app.route("/report")
def report_page():
    return render_template("report.html")


@app.route("/api/report")
def api_report():
    force = request.args.get("force", "false").lower() == "true"
    try:
        data = build_report(force=force)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info("Starting Flask server on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)


@app.route("/api/options/<ticker>")
def api_options(ticker: str):
    """Return options chain summary for a single ticker."""
    import yfinance as yf
    import numpy as np
    try:
        t = yf.Ticker(ticker.upper())
        exps = t.options
        if not exps:
            return jsonify({"error": "No options data available"}), 404

        # Use nearest 3 expirations
        rows = []
        for exp in exps[:3]:
            chain = t.option_chain(exp)
            calls = chain.calls
            puts  = chain.puts

            call_vol = int(calls["volume"].fillna(0).sum())
            put_vol  = int(puts["volume"].fillna(0).sum())
            call_oi  = int(calls["openInterest"].fillna(0).sum())
            put_oi   = int(puts["openInterest"].fillna(0).sum())
            avg_iv   = float(calls["impliedVolatility"].replace([np.inf], np.nan).dropna().mean() * 100) if len(calls) else 0

            # Top calls by volume
            top_calls = calls.nlargest(3, "volume")[["strike","lastPrice","volume","openInterest","impliedVolatility"]].fillna(0)
            top_puts  = puts.nlargest(3, "volume")[["strike","lastPrice","volume","openInterest","impliedVolatility"]].fillna(0)

            rows.append({
                "expiry":    exp,
                "call_vol":  call_vol,
                "put_vol":   put_vol,
                "call_oi":   call_oi,
                "put_oi":    put_oi,
                "cpr":       round(call_vol / max(put_vol, 1), 2),
                "avg_iv":    round(avg_iv, 1),
                "unusual":   bool(call_vol > put_vol * 2),
                "top_calls": top_calls.to_dict("records"),
                "top_puts":  top_puts.to_dict("records"),
            })

        price = float(t.history(period="1d")["Close"].iloc[-1]) if len(t.history(period="1d")) else 0

        return jsonify({"ticker": ticker.upper(), "price": price, "expirations": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
