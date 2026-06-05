# ─────────────────────────────────────────────
#  app.py  –  Flask web server
# ─────────────────────────────────────────────
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datetime, json, logging, threading, time
from flask import Flask, jsonify, render_template, request

from config import ALERT_CONFIG, HISTORY_CONFIG, POSITION_SIZING, PRICE_ALERTS_FILE, SCAN_INTERVAL_SECONDS
from universe import get_universe, UNIVERSE_MODES
from market_data import get_index_data, get_stock_data, get_sector_stats, get_ticker_detail, get_earnings_calendar
from history import get_global_store
import regime, technical, fundamental, sentiment, structural, risk, setups, scoring, validation
from main import score_ticker, should_alert, is_under20_popper
from options_engine import generate_options_plays

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("app")

app = Flask(__name__, template_folder=".")

# ── Shared state ──────────────────────────────────────────────────────────────
_lock           = threading.Lock()
_results        = []
_alerts         = []
_under20        = []
_options_plays  = []
_regime         = {"label": "unknown", "score": 0.0}
_index_snapshot = {}   # SPY/QQQ/IWM/VIX for Market Pulse
_sector_scores  = {}   # sector → avg upside score
_prev_sector_scores = {}
_scan_time      = None
_is_scanning    = False
_scan_progress  = {"current": 0, "total": 0, "ticker": ""}
_scan_stats     = {"duration_s": None, "processed": 0, "skipped": 0, "mode": ""}
_next_scan_time = None
_scan_count     = 0
_error_count    = 0
_universe_mode  = "sp500_top100"
_watchlist      = []   # server-side watchlist
_prev_alert_tickers: set = set()


def _build_sector_scores(results):
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in results:
        s = r.get("sector", "Unknown")
        buckets[s].append(r.get("upside", 0))
    return {s: round(sum(v)/len(v), 3) for s, v in buckets.items() if v}


def _build_index_snapshot(index_data):
    if not index_data:
        return {}
    spy = index_data.get("spy_prices", [])
    qqq = index_data.get("qqq_prices", [])
    iwm = index_data.get("iwm_prices", [])
    def chg(arr):
        if len(arr) > 1:
            return round((float(arr[-1]) - float(arr[-2])) / float(arr[-2]) * 100, 2)
        return 0.0
    return {
        "spy_price": round(float(spy[-1]), 2) if len(spy) else 0,
        "qqq_price": round(float(qqq[-1]), 2) if len(qqq) else 0,
        "iwm_price": round(float(iwm[-1]), 2) if len(iwm) else 0,
        "spy_chg":   chg(spy),
        "qqq_chg":   chg(qqq),
        "iwm_chg":   chg(iwm),
        "vix":       round(index_data.get("vix_current", 20), 2),
        "vix_chg":   round(index_data.get("vix_current", 20) - index_data.get("vix_20d_avg", 20), 2),
        "breadth":   round(index_data.get("breadth_pct_above_50ma", 0.5) * 100, 1),
    }


def run_scan_background():
    global _results, _alerts, _under20, _options_plays, _regime, _index_snapshot
    global _sector_scores, _prev_sector_scores, _scan_time, _is_scanning
    global _scan_progress, _scan_stats, _scan_count, _error_count
    global _next_scan_time, _prev_alert_tickers

    with _lock:
        _is_scanning   = True
        _scan_progress = {"current": 0, "total": 0, "ticker": ""}

    t0 = time.monotonic()
    _scan_count += 1
    mode = _universe_mode

    logger.info("Scan #%d starting (mode=%s)", _scan_count, mode)

    try:
        history_store = get_global_store()

        try:
            index_data = get_index_data()
        except Exception as e:
            logger.error("Index data fetch failed: %s", e)
            raise

        rl, rs = regime.compute_market_context(index_data, history_store)
        logger.info("Regime: %s %.3f", rl, rs)

        tickers = get_universe(mode=mode, watchlist=list(_watchlist))
        with _lock:
            _scan_progress["total"] = len(tickers)

        results, alerts, under20 = [], [], []
        skipped   = 0
        prev_set  = set(_prev_alert_tickers)

        for i, ticker in enumerate(tickers):
            if not ticker or len(ticker) > 10:
                skipped += 1
                continue

            with _lock:
                _scan_progress["current"] = i + 1
                _scan_progress["ticker"]  = ticker

            try:
                result = score_ticker(ticker, rl, rs, history_store, prev_alert_set=prev_set)
            except Exception as e:
                logger.warning("score_ticker failed %s: %s", ticker, e)
                skipped += 1
                continue

            if result is None:
                skipped += 1
                continue

            results.append(result)

            if should_alert(rs, result["upside"], result["risk"],
                            result["setup_score"], result["upside_change"], ALERT_CONFIG):
                alerts.append(result)
                history_store.log_alert(result)

            if is_under20_popper(result):
                under20.append(result)

        results.sort(key=lambda x: x["upside"], reverse=True)
        alerts.sort(key=lambda x: x["upside"], reverse=True)
        under20.sort(key=lambda x: x["upside"] * x["factor_scores"].get("technical", 0), reverse=True)

        # Options plays (async-ish — use top results)
        try:
            opt_plays = generate_options_plays(results)
        except Exception as e:
            logger.warning("Options plays failed: %s", e)
            opt_plays = []

        new_sectors = _build_sector_scores(results)
        snapshot    = _build_index_snapshot(index_data)
        dur         = round(time.monotonic() - t0, 2)

        history_store.save()
        logger.info("Scan done: %d ok, %d skipped, %d alerts, %.1fs", len(results), skipped, len(alerts), dur)

        with _lock:
            _prev_sector_scores = dict(_sector_scores)
            _results            = results
            _alerts             = alerts
            _under20            = under20
            _options_plays      = opt_plays
            _regime             = {"label": rl, "score": round(rs, 3)}
            _index_snapshot     = snapshot
            _sector_scores      = new_sectors
            _scan_time          = datetime.datetime.utcnow().isoformat() + "Z"
            _scan_stats         = {"duration_s": dur, "processed": len(results),
                                   "skipped": skipped, "mode": mode}
            _prev_alert_tickers = {a["ticker"] for a in alerts}
            _error_count        = 0

    except Exception as e:
        logger.error("Scan failed: %s", e, exc_info=True)
        with _lock:
            _error_count += 1
    finally:
        with _lock:
            _is_scanning = False
        _schedule_next()


def _schedule_next():
    global _next_scan_time
    _next_scan_time = datetime.datetime.utcnow() + datetime.timedelta(seconds=SCAN_INTERVAL_SECONDS)
    t = threading.Timer(SCAN_INTERVAL_SECONDS, run_scan_background)
    t.daemon = True
    t.start()


# Startup scan
threading.Thread(target=run_scan_background, daemon=True).start()


# ── Price alerts ──────────────────────────────────────────────────────────────

def _load_price_alerts():
    if not os.path.exists(PRICE_ALERTS_FILE):
        return []
    try:
        with open(PRICE_ALERTS_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def _save_price_alerts(alerts_list):
    with open(PRICE_ALERTS_FILE, "w") as f:
        json.dump(alerts_list, f)


def _check_price_alerts(results):
    """Fire notifications for triggered price alerts."""
    from notifier import send_alert as notify
    pa = _load_price_alerts()
    price_map = {r["ticker"]: r.get("last_price", 0) for r in results}
    for alert in pa:
        ticker    = alert.get("ticker", "")
        condition = alert.get("condition", "above")
        target    = float(alert.get("price", 0))
        current   = price_map.get(ticker, 0)
        if not current:
            continue
        triggered = (condition == "above" and current >= target) or \
                    (condition == "below"  and current <= target)
        if triggered and not alert.get("fired"):
            logger.info("Price alert triggered: %s %s $%.2f (current $%.2f)",
                        ticker, condition, target, current)
            alert["fired"] = True
            notify({
                "ticker":  ticker,
                "upside":  0,
                "risk":    0,
                "regime":  _regime.get("label", ""),
                "setups":  [],
                "factor_scores": {},
                "last_price": current,
                "is_new":  False,
                "upside_change": 0,
            })
    _save_price_alerts(pa)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/scan", methods=["POST"])
def trigger_scan():
    body = request.get_json(silent=True) or {}
    mode = body.get("mode")
    if mode:
        global _universe_mode
        _universe_mode = mode
    with _lock:
        if _is_scanning:
            return jsonify({"status": "already_running"})
    threading.Thread(target=run_scan_background, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/results")
def api_results():
    with _lock:
        nxt = _next_scan_time.isoformat() + "Z" if _next_scan_time else None

        # Sector rotation deltas
        sector_deltas = {}
        for s, v in _sector_scores.items():
            prev = _prev_sector_scores.get(s, v)
            sector_deltas[s] = round(v - prev, 3)

        # Pre/after market movers
        pre_movers  = sorted(
            [r for r in _results if abs(r.get("pre_market_chg",  0)) > 0.01],
            key=lambda x: abs(x.get("pre_market_chg", 0)), reverse=True)[:5]
        post_movers = sorted(
            [r for r in _results if abs(r.get("post_market_chg", 0)) > 0.01],
            key=lambda x: abs(x.get("post_market_chg", 0)), reverse=True)[:5]

        return jsonify({
            "regime":          _regime,
            "index_snapshot":  _index_snapshot,
            "scan_time":       _scan_time,
            "next_scan_time":  nxt,
            "is_scanning":     _is_scanning,
            "progress":        dict(_scan_progress),
            "total":           len(_results),
            "alerts":          _alerts[:20],
            "under20":         _under20[:20],
            "top":             _results,
            "scan_stats":      dict(_scan_stats),
            "sector_scores":   _sector_scores,
            "sector_deltas":   sector_deltas,
            "pre_movers":      pre_movers,
            "post_movers":     post_movers,
            "error_count":     _error_count,
            "universe_mode":   _universe_mode,
            "universe_modes":  UNIVERSE_MODES,
        })


@app.route("/api/options/plays")
def api_options_plays():
    with _lock:
        return jsonify({"plays": _options_plays})


@app.route("/api/options/<ticker>")
def api_options_chain(ticker):
    import yfinance as yf
    import numpy as np
    try:
        t    = yf.Ticker(ticker.upper())
        exps = t.options
        if not exps:
            return jsonify({"error": "No options data"}), 404

        rows = []
        for exp in exps[:3]:
            chain     = t.option_chain(exp)
            calls     = chain.calls
            puts      = chain.puts
            call_vol  = int(calls["volume"].fillna(0).sum())
            put_vol   = int(puts["volume"].fillna(0).sum())
            avg_iv    = float(calls["impliedVolatility"].replace([np.inf], np.nan).dropna().mean() * 100) if len(calls) else 0
            top_calls = calls.nlargest(3, "volume")[["strike","lastPrice","volume","openInterest","impliedVolatility"]].fillna(0)
            top_puts  = puts.nlargest(3,  "volume")[["strike","lastPrice","volume","openInterest","impliedVolatility"]].fillna(0)
            rows.append({
                "expiry":    exp,
                "call_vol":  call_vol,
                "put_vol":   put_vol,
                "call_oi":   int(calls["openInterest"].fillna(0).sum()),
                "put_oi":    int(puts["openInterest"].fillna(0).sum()),
                "cpr":       round(call_vol / max(put_vol, 1), 2),
                "avg_iv":    round(avg_iv, 1),
                "unusual":   bool(call_vol > put_vol * 2),
                "top_calls": top_calls.to_dict("records"),
                "top_puts":  top_puts.to_dict("records"),
            })

        hist  = t.history(period="1d")
        price = float(hist["Close"].iloc[-1]) if len(hist) else 0
        return jsonify({"ticker": ticker.upper(), "price": price, "expirations": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ticker/<ticker>")
def api_ticker_detail(ticker):
    with _lock:
        result = next((r for r in _results if r["ticker"] == ticker.upper()), None)
        sparkline = result.get("sparkline", []) if result else []
    try:
        detail = get_ticker_detail(ticker.upper())
        detail["sparkline"] = sparkline
        if result:
            detail["upside"]       = result.get("upside")
            detail["risk"]         = result.get("risk")
            detail["factor_scores"]= result.get("factor_scores")
            detail["setups"]       = result.get("setups")
            detail["narrative"]    = result.get("narrative")
            detail["position_size"]= result.get("position_size")
        return jsonify(detail)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/earnings")
def api_earnings():
    with _lock:
        tickers = [r["ticker"] for r in _results[:80]]
    try:
        cal = get_earnings_calendar(tickers, days_ahead=14)
        # Annotate with alert status
        alert_set = {a["ticker"] for a in _alerts}
        for item in cal:
            item["has_alert"] = item["ticker"] in alert_set
        return jsonify({"earnings": cal})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/heatmap")
def api_heatmap():
    with _lock:
        return jsonify({
            "sector_scores": _sector_scores,
            "sector_deltas": {
                s: round(_sector_scores.get(s, 0) - _prev_sector_scores.get(s, _sector_scores.get(s, 0)), 3)
                for s in _sector_scores
            },
        })


@app.route("/api/alerts/history")
def api_alert_history():
    limit = int(request.args.get("limit", 200))
    store = get_global_store()
    return jsonify({"history": store.get_alert_history(limit=limit)})


@app.route("/api/watchlist", methods=["GET"])
def get_watchlist():
    return jsonify({"watchlist": _watchlist})


@app.route("/api/watchlist", methods=["POST"])
def add_watchlist():
    global _watchlist
    ticker = (request.get_json(silent=True) or {}).get("ticker", "").upper().strip()
    if ticker and ticker not in _watchlist:
        _watchlist.append(ticker)
    return jsonify({"watchlist": _watchlist})


@app.route("/api/watchlist/<ticker>", methods=["DELETE"])
def remove_watchlist(ticker):
    global _watchlist
    _watchlist = [t for t in _watchlist if t != ticker.upper()]
    return jsonify({"watchlist": _watchlist})


@app.route("/api/price-alerts", methods=["GET"])
def get_price_alerts():
    return jsonify({"alerts": _load_price_alerts()})


@app.route("/api/price-alerts", methods=["POST"])
def add_price_alert():
    body   = request.get_json(silent=True) or {}
    ticker = body.get("ticker","").upper().strip()
    cond   = body.get("condition","above")
    price  = float(body.get("price", 0))
    if not ticker or not price:
        return jsonify({"error": "ticker and price required"}), 400
    pa = _load_price_alerts()
    pa.append({"ticker": ticker, "condition": cond, "price": price, "fired": False})
    _save_price_alerts(pa)
    return jsonify({"alerts": pa})


@app.route("/api/price-alerts/<ticker>", methods=["DELETE"])
def delete_price_alert(ticker):
    pa = [a for a in _load_price_alerts() if a.get("ticker") != ticker.upper()]
    _save_price_alerts(pa)
    return jsonify({"alerts": pa})


@app.route("/api/settings", methods=["GET"])
def get_settings():
    from config import ALERT_CONFIG, POSITION_SIZING, SCAN_INTERVAL_SECONDS
    return jsonify({
        "alert_config":      ALERT_CONFIG,
        "position_sizing":   POSITION_SIZING,
        "scan_interval":     SCAN_INTERVAL_SECONDS,
        "universe_mode":     _universe_mode,
        "universe_modes":    UNIVERSE_MODES,
        "telegram_enabled":  bool(os.environ.get("TELEGRAM_BOT_TOKEN")),
        "email_enabled":     os.environ.get("EMAIL_ENABLED","false") == "true",
    })


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """Update runtime settings (account size, thresholds, universe mode)."""
    global _universe_mode
    import config
    body = request.get_json(silent=True) or {}

    if "universe_mode" in body:
        _universe_mode = body["universe_mode"]
    if "account_size" in body:
        config.POSITION_SIZING["account_size"] = float(body["account_size"])
    if "risk_per_trade_pct" in body:
        config.POSITION_SIZING["risk_per_trade_pct"] = float(body["risk_per_trade_pct"])
    if "upside_threshold" in body:
        config.ALERT_CONFIG["upside_percentile_threshold"] = float(body["upside_threshold"])
    if "risk_max" in body:
        config.ALERT_CONFIG["risk_percentile_max"] = float(body["risk_max"])
    if "scan_interval" in body:
        config.SCAN_INTERVAL_SECONDS = int(body["scan_interval"])

    return jsonify({"status": "updated"})


@app.route("/api/health")
def api_health():
    with _lock:
        return jsonify({
            "status":        "scanning" if _is_scanning else "idle",
            "last_scan":     _scan_time,
            "scan_count":    _scan_count,
            "error_count":   _error_count,
            "tickers_last":  _scan_stats.get("processed", 0),
            "alerts_last":   len(_alerts),
            "mode":          _universe_mode,
            "uptime_ok":     _error_count < 5,
        })


@app.route("/api/export/csv")
def export_csv():
    import io, csv
    from flask import Response
    with _lock:
        rows = list(_results)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Ticker","Sector","Price","Upside","Risk","Technical",
                     "Fundamental","Sentiment","Structural","Setup Score","Setups",
                     "Beta","Earnings Date","52W High","52W Low"])
    for r in rows:
        f = r.get("factor_scores", {})
        writer.writerow([
            r.get("ticker"),      r.get("sector"),
            r.get("last_price"),  round(r.get("upside", 0), 3),
            round(r.get("risk", 0), 3),
            round(f.get("technical",   0), 3),
            round(f.get("fundamental", 0), 3),
            round(f.get("sentiment",   0), 3),
            round(f.get("structural",  0), 3),
            round(r.get("setup_score", 0), 3),
            "|".join(r.get("setups", [])),
            round(r.get("beta", 1), 2),
            r.get("earnings_date", ""),
            r.get("52w_high", ""),  r.get("52w_low", ""),
        ])
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=stockbot_results.csv"},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info("Starting on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
