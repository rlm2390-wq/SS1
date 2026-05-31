# ─────────────────────────────────────────────
#  app.py  –  Flask web server
# ─────────────────────────────────────────────
from __future__ import annotations
import sys
import os

# Ensure project root is on sys.path regardless of where gunicorn launches from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datetime
import threading
from flask import Flask, render_template, jsonify

from config import ALERT_CONFIG, HISTORY_CONFIG
from data.universe import get_universe
from data.market_data import get_index_data, get_stock_data, get_sector_stats
from storage.history import HistoryStore
from brains import regime, technical, fundamental, sentiment, structural, risk, setups, scoring, validation
from main import score_ticker, should_alert

app = Flask(__name__)

# ── Shared scan state ─────────────────────────────────────────────────────────
_scan_lock      = threading.Lock()
_last_results   = []
_last_alerts    = []
_last_regime    = {"label": "unknown", "score": 0.0}
_last_scan_time = None
_is_scanning    = False
_scan_progress  = {"current": 0, "total": 0, "ticker": ""}


def run_scan_background():
    global _last_results, _last_alerts, _last_regime, _last_scan_time
    global _is_scanning, _scan_progress

    with _scan_lock:
        _is_scanning = True
        _scan_progress = {"current": 0, "total": 0, "ticker": ""}

    try:
        history_store = HistoryStore()

        # Market context
        index_data = get_index_data()
        rl, rs     = regime.compute_market_context(index_data, history_store)

        tickers = get_universe()
        with _scan_lock:
            _scan_progress["total"] = len(tickers)

        results = []
        alerts  = []

        for i, ticker in enumerate(tickers):
            with _scan_lock:
                _scan_progress["current"] = i + 1
                _scan_progress["ticker"]  = ticker

            result = score_ticker(ticker, rl, rs, history_store)
            if result is None:
                continue
            results.append(result)
            if should_alert(rs, result["upside"], result["risk"],
                            result["setup_score"], result["upside_change"], ALERT_CONFIG):
                alerts.append(result)

        results.sort(key=lambda x: x["upside"], reverse=True)
        alerts.sort(key=lambda x: x["upside"], reverse=True)

        with _scan_lock:
            _last_results   = results
            _last_alerts    = alerts
            _last_regime    = {"label": rl, "score": round(rs, 3)}
            _last_scan_time = datetime.datetime.utcnow().isoformat() + "Z"

    except Exception as e:
        print(f"[scan error] {e}")
    finally:
        with _scan_lock:
            _is_scanning = False


# Run an initial scan on startup
threading.Thread(target=run_scan_background, daemon=True).start()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/scan", methods=["POST"])
def trigger_scan():
    with _scan_lock:
        if _is_scanning:
            return jsonify({"status": "already_running"})
    threading.Thread(target=run_scan_background, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/results")
def api_results():
    with _scan_lock:
        return jsonify({
            "regime":      _last_regime,
            "scan_time":   _last_scan_time,
            "is_scanning": _is_scanning,
            "progress":    dict(_scan_progress),
            "total":       len(_last_results),
            "alerts":      _last_alerts[:20],
            "top":         _last_results[:50],
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
