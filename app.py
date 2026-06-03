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
from main import score_ticker, should_alert

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("app")

app = Flask(__name__, template_folder=".")

# ── Shared scan state ─────────────────────────────────────────────────────────
_scan_lock            = threading.Lock()
_last_results         = []
_last_alerts          = []
_last_regime          = {"label": "unknown", "score": 0.0}
_last_scan_time       = None
_is_scanning          = False
_scan_progress        = {"current": 0, "total": 0, "ticker": ""}
_last_scan_stats      = {"duration_s": None, "tickers_processed": 0, "tickers_skipped": 0}
_next_scan_time       = None

# Full scan of the entire universe every 10 minutes.
_SCAN_INTERVAL        = 600    # 10 minutes


def schedule_next_scan():
    """Schedule the next full scan in 10 minutes."""
    global _next_scan_time
    _next_scan_time = datetime.datetime.utcnow() + datetime.timedelta(seconds=_SCAN_INTERVAL)
    logger.info("Next scheduled scan at %s UTC", _next_scan_time.strftime("%Y-%m-%dT%H:%M:%S"))
    t = threading.Timer(_SCAN_INTERVAL, run_scan_background)
    t.daemon = True
    t.start()


def run_scan_background():
    global _last_results, _last_alerts, _last_regime, _last_scan_time
    global _is_scanning, _scan_progress, _last_scan_stats

    with _scan_lock:
        _is_scanning = True
        _scan_progress = {"current": 0, "total": 0, "ticker": ""}

    scan_start = time.monotonic()
    logger.info("Scan started")

    try:
        history_store = HistoryStore()

        # Market context
        try:
            index_data = get_index_data()
        except Exception as exc:
            logger.error("Failed to fetch index data: %s", exc, exc_info=True)
            raise

        rl, rs = regime.compute_market_context(index_data, history_store)
        logger.info("Market regime: %s (score=%.3f)", rl, rs)

        # Always scan the full universe (S&P 500 + IPOs + drops)
        tickers = get_universe()
        logger.info("Scan universe: %d tickers", len(tickers))

        with _scan_lock:
            _scan_progress["total"] = len(tickers)

        results  = []
        alerts   = []
        skipped  = 0

        for i, ticker in enumerate(tickers):
            # Skip obviously invalid ticker strings before hitting the API
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

        results.sort(key=lambda x: x["upside"], reverse=True)
        alerts.sort(key=lambda x: x["upside"], reverse=True)

        scan_duration = round(time.monotonic() - scan_start, 2)
        logger.info(
            "Scan complete: %d processed, %d skipped, %d alerts, %.1fs",
            len(results), skipped, len(alerts), scan_duration,
        )

        with _scan_lock:
            _last_results    = results
            _last_alerts     = alerts
            _last_regime     = {"label": rl, "score": round(rs, 3)}
            _last_scan_time  = datetime.datetime.utcnow().isoformat() + "Z"
            _last_scan_stats = {
                "duration_s":        scan_duration,
                "tickers_processed": len(results),
                "tickers_skipped":   skipped,
            }

    except Exception as exc:
        logger.error("Scan failed with unhandled exception: %s", exc, exc_info=True)
    finally:
        with _scan_lock:
            _is_scanning = False
        # Schedule the next recurring scan regardless of success/failure
        schedule_next_scan()


# Run an initial scan on startup, then recurring scans every 10 minutes
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
        next_scan_iso = (
            _next_scan_time.isoformat() + "Z" if _next_scan_time else None
        )
        return jsonify({
            "regime":          _last_regime,
            "scan_time":       _last_scan_time,
            "next_scan_time":  next_scan_iso,
            "is_scanning":     _is_scanning,
            "progress":        dict(_scan_progress),
            "total":           len(_last_results),
            "alerts":          _last_alerts[:20],
            "top":             _last_results,
            "scan_stats":      dict(_last_scan_stats),
        })


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    port = int(os.environ.get("PORT", 8080))
    logger.info("Starting Flask server on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
