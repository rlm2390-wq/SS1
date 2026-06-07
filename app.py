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
from history import HistoryStore
import regime, technical, fundamental, sentiment, structural, risk, setups, scoring, validation
from main import score_ticker, should_alert, is_under20_popper, is_under10_popper
from under10 import filter_and_rank_under10
from scanners import run_all_scanners
from scoring import get_top2_weights, get_current_weights

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
        history_store = HistoryStore()

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
                            result["setup_score"], result["upside_change"], ALERT_CONFIG):
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

        with _scan_lock:
            _last_results    = results
            _last_alerts     = alerts
            _last_under20    = under20
            _last_pre_signals = pre_signals_list[:25]
            _last_scanners    = scanner_results
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
            "scan_stats":     dict(_last_scan_stats),
        })


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
