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
import os as _os
import time
from flask import Flask, render_template, jsonify, request

from config import ALERT_CONFIG, HISTORY_CONFIG, UNIVERSE_CONFIG
from universe import get_universe
from market_data import get_index_data, get_stock_data, get_sector_stats
from history import HistoryStore, get_global_store
import regime, technical, fundamental, sentiment, structural, risk, setups, scoring, validation
from main import score_ticker, should_alert, is_under10_popper
from under10 import filter_and_rank_under10
from scanners import run_all_scanners
from scoring import get_top2_weights, get_current_weights
from pre_market import start_premarket_thread, get_premarket_results, is_premarket, premarket_status
from scheduler import start_scheduler_thread
from dashboard_backend import get_dashboard_state, get_setup_card as db_get_setup_card, log_trade_from_setup
import cache as _cache_module
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

import json as _json, math as _math2
class _SafeEncoder(_json.JSONEncoder):
    """Converts NaN/Inf to None and numpy types to Python natives globally."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):  return obj.tolist()
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating):
                v = float(obj)
                return None if (_math2.isnan(v) or _math2.isinf(v)) else v
            if isinstance(obj, np.bool_):    return bool(obj)
        except ImportError:
            pass
        return super().default(obj)
    def iterencode(self, o, _one_shot=False):
        def _fix(v):
            if isinstance(v, float) and (_math2.isnan(v) or _math2.isinf(v)):
                return None
            if isinstance(v, dict):  return {kk: _fix(vv) for kk, vv in v.items()}
            if isinstance(v, list):  return [_fix(i) for i in v]
            return v
        return super().iterencode(_fix(o), _one_shot)

app.json_encoder = _SafeEncoder

# ── Shared scan state ─────────────────────────────────────────────────────────
_scan_lock       = threading.Lock()
_last_results    = []
_last_alerts     = []
_last_under20    = []
_last_pre_signals = []   # tickers with 1+ pre-signals
_last_scanners    = {}   # six discovery scanners
_last_index_data  = {}   # SPY/QQQ/IWM/VIX/Breadth snapshot
_last_sector_scores = {}  # sector UpsideScore averages
_last_sector_deltas = {}  # sector score changes
_weight_display   = ""   # e.g. "Tech 34% · Fund 28%"
_watchlist         = []   # server-side watchlist
_last_regime     = {"label": "unknown", "score": 0.0}
_last_scan_time  = None
# ── Scan state machine ────────────────────────────────────────────────────────
# IDLE → RUNNING → COMPLETE → IDLE
# Manual scans are queued, never silently dropped.
_is_scanning      = False
_scan_queued      = False   # True = a manual scan is waiting for current to finish
_scan_progress   = {"current": 0, "total": 0, "ticker": ""}
_last_scan_stats = {"duration_s": None, "tickers_processed": 0, "tickers_skipped": 0, "scan_mode": ""}
_next_scan_time  = None

_SCAN_INTERVAL       = 600   # 10 minutes (quick scan cadence)
_scan_count          = 0     # incremented each cycle; every 6th = full scan


def schedule_next_scan():
    global _next_scan_time
    _next_scan_time = datetime.datetime.utcnow() + datetime.timedelta(seconds=_SCAN_INTERVAL)
    logger.info("Next scheduled scan at %s UTC", _next_scan_time.strftime("%Y-%m-%dT%H:%M:%S"))
    t = threading.Timer(_SCAN_INTERVAL, run_scan_background)
    t.daemon = True
    t.start()


def run_scan_background():
    global _is_scanning, _scan_queued, _scan_progress
    global _last_results, _last_alerts, _last_under20, _last_regime, _last_scan_time
    global _last_scan_stats, _last_pre_signals, _weight_display, _scan_count
    global _last_scanners, _last_index_data, _last_sector_scores, _last_sector_deltas

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
            # Sanitize numpy types and NaN/Inf so index_data is JSON-serializable
            import math as _math
            def _sanitize(obj):
                if obj is None: return None
                if isinstance(obj, dict):
                    return {k: _sanitize(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_sanitize(v) for v in obj]
                try:
                    import numpy as np
                    if isinstance(obj, np.ndarray):
                        return [_sanitize(v) for v in obj.tolist()]
                    if isinstance(obj, (np.integer,)):
                        return int(obj)
                    if isinstance(obj, (np.floating,)):
                        v = float(obj)
                        return None if (_math.isnan(v) or _math.isinf(v)) else v
                    if isinstance(obj, (np.bool_,)):
                        return bool(obj)
                except ImportError:
                    pass
                if isinstance(obj, float):
                    return None if (_math.isnan(obj) or _math.isinf(obj)) else obj
                return obj
            with _scan_lock:
                _last_index_data = _sanitize(index_data or {})
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
            _last_regime     = {"label": rl, "score": round(rs, 3)}
            _last_scan_time  = datetime.datetime.utcnow().isoformat() + "Z"
            _last_scan_stats = {
                "duration_s":        scan_duration,
                "tickers_processed": len(results),
                "processed":         len(results),
                "tickers_skipped":   skipped,
                "mode":              scan_mode,
                "scan_mode":         scan_mode,
            }
            # Compute sector scores from results
            _sec_scores: dict = {}
            for r in results:
                sec = r.get("sector") or "Unknown"
                if sec not in _sec_scores:
                    _sec_scores[sec] = []
                _sec_scores[sec].append(r.get("upside", 0))
            _last_sector_scores = {s: round(sum(v)/len(v), 3) for s, v in _sec_scores.items()}
            _last_sector_deltas = {}  # delta tracking requires prior scan history

    except Exception as exc:
        logger.error("Scan failed: %s", exc, exc_info=True)
    finally:
        with _scan_lock:
            global _scan_queued
            _is_scanning = False
            queued = _scan_queued
            _scan_queued = False
        schedule_next_scan()
        # If a manual scan was queued while we were running, honor it now
        if queued:
            logger.info("Running queued manual scan")
            threading.Thread(target=run_scan_background, daemon=True).start()


# ── Single-worker startup guard ──────────────────────────────────────────────
# Railway runs 2 gunicorn workers — use an exclusive file lock so only
# worker 1 starts the background scan and pre-market threads.
# Keep file handle at module level so the lock isn't released by GC
_primary_lock_fh = None

def _try_become_primary_worker():
    """Return True if this worker wins the startup lock."""
    global _primary_lock_fh
    import fcntl
    lock_path = "/tmp/stockbot_primary.lock"
    try:
        fh = open(lock_path, "w")
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.write(str(_os.getpid()))
        fh.flush()
        _primary_lock_fh = fh   # keep alive — releasing fh releases lock
        return True
    except (IOError, OSError):
        return False

_is_primary = _try_become_primary_worker()

if _is_primary:
    threading.Thread(target=run_scan_background, daemon=True).start()
    start_premarket_thread(
        get_watchlist_fn     = lambda: list(_watchlist),
        get_recent_alerts_fn = lambda: [a["ticker"] for a in _last_alerts],
        get_all_results_fn   = lambda: list(_last_results),
    )
    start_scheduler_thread()   # Webull real-time section scheduler
    logger.info("Primary worker started — scan + pre-market + Webull scheduler launched")
else:
    logger.info("Secondary worker started — scan threads skipped")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/scan", methods=["POST"])
def trigger_scan():
    global _scan_queued
    data = request.get_json(silent=True) or {}
    mode = data.get("mode")
    if mode:
        UNIVERSE_CONFIG["mode"] = mode
        logger.info("Universe mode set to: %s", mode)
    with _scan_lock:
        if _is_scanning:
            _scan_queued = True
            logger.info("Scan running — manual scan queued for after completion")
            return jsonify({"status": "queued"})
    logger.info("Manual scan triggered via /api/scan")
    threading.Thread(target=run_scan_background, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/results")
def api_results():
    with _scan_lock:
        next_scan_iso = _next_scan_time.isoformat() + "Z" if _next_scan_time else None
        state = get_dashboard_state(
            last_results  = list(_last_results),
            last_alerts   = list(_last_alerts),
            last_under20  = list(_last_under20),
            last_regime   = dict(_last_regime),
            last_scanners = dict(_last_scanners),
            scan_stats    = dict(_last_scan_stats),
            index_data    = dict(_last_index_data),
        )
        state.update({
            "scan_time":      _last_scan_time,
            "next_scan_time": next_scan_iso,
            "is_scanning":    _is_scanning,
            "progress":       dict(_scan_progress),
            "pre_signals":    list(_last_pre_signals),
            "weight_display": _weight_display,
            "signal_summary": get_mini_summary(),
            "universe_mode":  UNIVERSE_CONFIG.get("mode", "sp500_top100"),
            "sector_scores":  _last_sector_scores,
            "sector_deltas":  _last_sector_deltas,
        })
        return jsonify(state)




@app.route("/api/premarket")
def api_premarket():
    """Return current pre-market alert state for both tiers."""
    data = get_premarket_results()
    data["is_premarket"] = is_premarket()
    return jsonify(data)


@app.route("/api/premarket/scan", methods=["POST"])
def trigger_premarket_scan():
    """Manually trigger a pre-market scan (for testing outside market hours)."""
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
    """Compute a full trade setup card using yfinance history + Webull live price."""
    ticker = ticker.upper()
    with _scan_lock:
        result = next((r for r in _last_results if r["ticker"] == ticker), None)
    if not result:
        return jsonify({"error": "Ticker not in last scan results"}), 404
    try:
        from scheduler import get_rt_cache
        rt_quotes = get_rt_cache().get("quotes", {})
        setup = db_get_setup_card(ticker, result, rt_quotes)
        return jsonify(setup)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/setup/<ticker>/trade", methods=["POST"])
def api_log_trade(ticker: str):
    """Log a trade from the Setup Card confirmation flow."""
    ticker = ticker.upper()
    data   = request.get_json(silent=True) or {}
    entry_price = float(data.get("entry_price", 0))
    if not entry_price:
        return jsonify({"error": "entry_price required"}), 400
    with _scan_lock:
        result = next((r for r in _last_results if r["ticker"] == ticker), None)
    if not result:
        return jsonify({"error": "Ticker not found"}), 404
    from scheduler import get_rt_cache
    setup = db_get_setup_card(ticker, result, get_rt_cache().get("quotes", {}))
    return jsonify(log_trade_from_setup(ticker, entry_price, setup))


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



@app.route("/api/rt/quotes")
def api_rt_quotes():
    """Return cached Webull real-time quotes for all tracked tickers."""
    try:
        from scheduler import get_rt_cache
        cache = get_rt_cache()
        return jsonify({
            "quotes":       cache.get("quotes", {}),
            "last_updated": cache.get("last_updated", {}),
            "rt_enabled":   bool(cache.get("quotes")),
        })
    except Exception as e:
        return jsonify({"quotes": {}, "rt_enabled": False, "error": str(e)})



@app.route("/api/cache/stats")
def api_cache_stats():
    """Cache health check."""
    return jsonify({
        "cache": _cache_module.stats(),
        "data_router": __import__("data_router").status(),
    })


@app.route("/api/cache/invalidate", methods=["POST"])
def api_cache_invalidate():
    """Force-invalidate specific cache keys. POST {prefix: "stock_data"} or {key: "universe:sp500_full"}."""
    data   = request.get_json(silent=True) or {}
    prefix = data.get("prefix")
    key    = data.get("key")
    if prefix:
        n = _cache_module.invalidate_prefix(prefix)
        return jsonify({"status": "ok", "removed": n, "prefix": prefix})
    if key:
        _cache_module.invalidate(key)
        return jsonify({"status": "ok", "key": key})
    # Full universe refresh
    from universe import get_sp500_tickers
    get_sp500_tickers(force_refresh=True)
    return jsonify({"status": "ok", "action": "universe_refreshed"})



@app.route("/api/debug/scanner")
def api_debug_scanner():
    """Show first result dict so we can verify scanner field structure."""
    with _scan_lock:
        if not _last_results:
            return jsonify({"error": "No results yet"})
        sample = _last_results[0]
        return jsonify({
            "ticker":           sample.get("ticker"),
            "last_price":       sample.get("last_price"),
            "risk":             sample.get("risk"),
            "pre_signals_keys": list((sample.get("pre_signals") or {}).keys()),
            "pre_signal_count": (sample.get("pre_signals") or {}).get("signal_count", 0),
            "microstructure_keys": list((sample.get("microstructure") or {}).keys()),
            "micro_sub_keys":   list(((sample.get("microstructure") or {}).get("sub_scores") or {}).keys()),
            "dollar_vol_accel": ((sample.get("microstructure") or {}).get("sub_scores") or {}).get("dollar_vol_accel"),
            "sub_scores_keys":  list((sample.get("sub_scores") or {}).keys()),
            "risk_sub_keys":    list(((sample.get("sub_scores") or {}).get("risk") or {}).keys()),
            "struct_sub_keys":  list(((sample.get("sub_scores") or {}).get("structural") or {}).keys()),
            "setups":           sample.get("setups"),
            "scanner_results":  {k: len(v) for k, v in _last_scanners.items()},
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
