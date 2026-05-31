# ─────────────────────────────────────────────
#  data/market_data.py  –  Real data via yfinance
#
#  Fetches live prices, fundamentals, short
#  interest, insider activity, and news from
#  Yahoo Finance. Falls back to safe defaults
#  if a field is missing so the bot never
#  crashes on incomplete data.
# ─────────────────────────────────────────────
from __future__ import annotations
import datetime
import numpy as np
from typing import Dict, Any

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(val, default=0.0):
    """Return val if it's a real number, otherwise default."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _moving_average(series: np.ndarray, window: int) -> np.ndarray:
    ma = np.full_like(series, np.nan)
    for i in range(window - 1, len(series)):
        ma[i] = series[i - window + 1 : i + 1].mean()
    return ma


def _rsi(prices: np.ndarray, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains  = deltas[deltas > 0].mean() if (deltas > 0).any() else 0.0
    losses = (-deltas[deltas < 0]).mean() if (deltas < 0).any() else 1e-9
    rs     = gains / (losses + 1e-9)
    return float(100 - 100 / (1 + rs))


def _bb_width(prices: np.ndarray, window: int = 20) -> float:
    if len(prices) < window:
        return 0.05
    slice_ = prices[-window:]
    mid    = slice_.mean()
    std    = slice_.std()
    return float(4 * std / (mid + 1e-9))


# ── Index data ────────────────────────────────────────────────────────────────

def get_index_data() -> Dict[str, Any]:
    """
    Fetch SPY, QQQ, IWM prices and VIX from Yahoo Finance.
    """
    if not YF_AVAILABLE:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    def _fetch(ticker, period="1y"):
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        return hist["Close"].values.astype(float)

    spy = _fetch("SPY")
    qqq = _fetch("QQQ")
    iwm = _fetch("IWM")
    vix = _fetch("^VIX")

    # Breadth proxy: use % of Dow 30 above their 50-day MA
    dow30 = ["AAPL","MSFT","JPM","JNJ","V","WMT","PG","UNH","HD","CVX",
             "MRK","AMGN","CAT","GS","MMM","MCD","AXP","IBM","HON","BA",
             "DIS","TRV","CRM","NKE","INTC","VZ","KO","DOW","WBA","CSCO"]
    above_50 = 0
    for sym in dow30[:15]:   # limit to 15 to avoid rate limits
        try:
            h = yf.Ticker(sym).history(period="3mo")["Close"].values
            ma = _moving_average(h, 50)
            if not np.isnan(ma[-1]) and h[-1] > ma[-1]:
                above_50 += 1
        except Exception:
            pass
    breadth = above_50 / 15.0

    # Put/call ratio from VIX as proxy (no free source; use VIX level)
    vix_cur = float(vix[-1]) if len(vix) else 20.0
    vix_avg = float(vix[-20:].mean()) if len(vix) >= 20 else vix_cur
    # Rough put/call proxy: high VIX → high P/C
    pc_proxy = float(np.clip(vix_cur / 20.0 * 0.85, 0.4, 2.0))

    return {
        "spy_prices":  spy,
        "qqq_prices":  qqq,
        "iwm_prices":  iwm,
        "spy_ma50":    _moving_average(spy, 50),
        "spy_ma200":   _moving_average(spy, 200),
        "qqq_ma50":    _moving_average(qqq, 50),
        "vix_series":  vix,
        "vix_current": vix_cur,
        "vix_20d_avg": vix_avg,
        "breadth_pct_above_50ma": breadth,
        "put_call_ratio": pc_proxy,
        "advance_decline": 1.0,   # placeholder; no free real-time source
    }


# ── Stock data ────────────────────────────────────────────────────────────────

def get_stock_data(ticker: str, lookback_days: int = 252) -> Dict[str, Any]:
    """
    Fetch all per-stock data needed by every brain.
    """
    if not YF_AVAILABLE:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    t    = yf.Ticker(ticker)
    hist = t.history(period="1y")
    info = t.info or {}

    if hist.empty or len(hist) < 20:
        return {}   # validation brain will reject this

    prices  = hist["Close"].values.astype(float)
    volumes = hist["Volume"].values.astype(float)

    # Trim to lookback_days
    if len(prices) > lookback_days:
        prices  = prices[-lookback_days:]
        volumes = volumes[-lookback_days:]

    last = float(prices[-1])
    ma20  = _moving_average(prices, 20)
    ma50  = _moving_average(prices, 50)
    ma200 = _moving_average(prices, 200)

    # ATR
    hi = hist["High"].values.astype(float)[-len(prices):]
    lo = hist["Low"].values.astype(float)[-len(prices):]
    if len(hi) > 1:
        tr  = np.maximum(hi[1:] - lo[1:],
              np.maximum(np.abs(hi[1:] - prices[:-1]),
                         np.abs(lo[1:] - prices[:-1])))
        atr14 = float(tr[-14:].mean()) if len(tr) >= 14 else float(tr.mean())
    else:
        atr14 = last * 0.02
    atr_pct = atr14 / (last + 1e-9)

    avg_dollar_vol = float((prices[-20:] * volumes[-20:]).mean())

    # ── Fundamentals ──────────────────────────────────────────────────────────
    fundamentals = {
        "revenue_yoy":       _safe(info.get("revenueGrowth"),       0.05),
        "eps_yoy":           _safe(info.get("earningsGrowth"),      0.05),
        "gross_margin":      _safe(info.get("grossMargins"),        0.40),
        "margin_trend":      _safe(info.get("operatingMargins"), 0.10) - 0.10,
        "peg":               _safe(info.get("pegRatio"),             2.0),
        "ev_sales":          _safe(info.get("enterpriseToRevenue"), 3.0),
        "debt_to_equity":    _safe(info.get("debtToEquity"), 50.0) / 100.0,
        "cash_runway_years": _safe(info.get("totalCash"), 0) /
                             max(_safe(info.get("totalRevenue"), 1), 1) * 2,
        "earnings_surprise": _safe(info.get("earningsQuarterlyGrowth"), 0.0),
    }

    # ── Short interest ─────────────────────────────────────────────────────────
    short_float_pct = _safe(info.get("shortPercentOfFloat"), 0.05) * 100
    shares_short    = _safe(info.get("sharesShort"),         0)
    avg_vol_10      = _safe(info.get("averageVolume10days"),
                            volumes[-10:].mean() if len(volumes) >= 10 else volumes.mean())
    days_to_cover   = shares_short / max(avg_vol_10, 1)

    short_interest = {
        "short_float_pct": short_float_pct,
        "days_to_cover":   float(np.clip(days_to_cover, 0, 30)),
        "borrow_cost_pct": float(np.clip(short_float_pct * 0.3, 0.1, 30.0)),
        "short_trend":     0.0,   # yfinance doesn't provide historical short data
    }

    # ── Insider activity ──────────────────────────────────────────────────────
    try:
        ins_df = t.insider_purchases
        if ins_df is not None and not ins_df.empty and "Shares" in ins_df.columns:
            buys  = len(ins_df[ins_df.get("Transaction", ins_df.columns[0]).str.contains("Buy|Purchase", na=False)])
            sells = max(0, len(ins_df) - buys)
            net   = float(ins_df["Value"].sum()) if "Value" in ins_df.columns else 0.0
        else:
            buys, sells, net = 0, 0, 0.0
    except Exception:
        buys, sells, net = 0, 0, 0.0

    insider_activity = {
        "buy_events_90d":  buys,
        "sell_events_90d": sells,
        "net_buy_usd_90d": net,
        "largest_buy_usd": abs(net) * 0.6,
    }

    # ── News ──────────────────────────────────────────────────────────────────
    news_items = []
    try:
        raw_news = t.news or []
        for n in raw_news[:10]:
            # yfinance news doesn't include sentiment — use title length as a
            # rough heuristic until you add an NLP library (e.g. transformers)
            title = n.get("title", "")
            positive_words = ["beat","surge","soar","growth","record","upgrade",
                              "raise","rally","strong","profit","bullish","buy"]
            negative_words = ["miss","fall","drop","cut","downgrade","loss",
                              "warn","decline","weak","bearish","sell","risk"]
            score = sum(1 for w in positive_words if w in title.lower()) \
                  - sum(1 for w in negative_words if w in title.lower())
            sentiment = float(np.clip(score / 3.0, -1.0, 1.0))
            news_items.append({
                "headline":     title,
                "sentiment":    sentiment,
                "published_at": datetime.date.today().isoformat(),
            })
    except Exception:
        pass

    # ── Options flow ──────────────────────────────────────────────────────────
    options_flow = {
        "call_vol_ratio":  1.0,
        "call_put_ratio":  1.0,
        "iv_percentile":   50.0,
        "unusual_options": False,
    }
    try:
        exp_dates = t.options
        if exp_dates:
            chain = t.option_chain(exp_dates[0])
            call_vol = float(chain.calls["volume"].sum())
            put_vol  = float(chain.puts["volume"].sum())
            cpr      = call_vol / max(put_vol, 1)
            options_flow = {
                "call_vol_ratio":  float(np.clip(cpr, 0.1, 5.0)),
                "call_put_ratio":  float(np.clip(cpr, 0.1, 5.0)),
                "iv_percentile":   float(np.clip(
                    chain.calls["impliedVolatility"].mean() * 100, 0, 100)),
                "unusual_options": bool(cpr > 2.5),
            }
    except Exception:
        pass

    return {
        "ticker":             ticker,
        "sector":             info.get("sector") or info.get("industry") or "Unknown",
        "prices":             prices,
        "volumes":            volumes,
        "ma20":               ma20,
        "ma50":               ma50,
        "ma200":              ma200,
        "rsi":                _rsi(prices),
        "atr_pct":            atr_pct,
        "bb_width":           _bb_width(prices),
        "avg_dollar_vol":     avg_dollar_vol,
        "fundamentals":       fundamentals,
        "short_interest":     short_interest,
        "insider_activity":   insider_activity,
        "news_items":         news_items,
        "options_flow":       options_flow,
        "float_shares_m":     _safe(info.get("floatShares"), 500e6) / 1e6,
        "inst_ownership_pct": _safe(info.get("heldPercentInstitutions"), 0.5) * 100,
        "analyst_upgrades_30d":      0,   # not available in free yfinance
        "analyst_downgrades_30d":    0,
        "analyst_target_change_pct": (
            _safe(info.get("targetMeanPrice"), last) / (last + 1e-9) - 1.0
        ),
    }


# ── Sector stats ──────────────────────────────────────────────────────────────

# Hardcoded sector medians — avoids an extra 500 API calls per scan.
# Update these periodically or replace with a live sector ETF fetch.
_SECTOR_MEDIANS = {
    "Technology":             {"ev_sales_median": 6.0,  "peg_median": 2.0, "rev_growth_median": 0.12, "gross_margin_median": 0.62},
    "Healthcare":             {"ev_sales_median": 3.5,  "peg_median": 2.2, "rev_growth_median": 0.08, "gross_margin_median": 0.55},
    "Financials":             {"ev_sales_median": 2.5,  "peg_median": 1.5, "rev_growth_median": 0.07, "gross_margin_median": 0.45},
    "Consumer Cyclical":      {"ev_sales_median": 1.8,  "peg_median": 1.8, "rev_growth_median": 0.06, "gross_margin_median": 0.38},
    "Consumer Defensive":     {"ev_sales_median": 1.5,  "peg_median": 2.0, "rev_growth_median": 0.04, "gross_margin_median": 0.35},
    "Energy":                 {"ev_sales_median": 1.2,  "peg_median": 1.2, "rev_growth_median": 0.05, "gross_margin_median": 0.28},
    "Industrials":            {"ev_sales_median": 2.0,  "peg_median": 1.8, "rev_growth_median": 0.06, "gross_margin_median": 0.33},
    "Materials":              {"ev_sales_median": 1.8,  "peg_median": 1.5, "rev_growth_median": 0.05, "gross_margin_median": 0.30},
    "Utilities":              {"ev_sales_median": 2.2,  "peg_median": 2.5, "rev_growth_median": 0.03, "gross_margin_median": 0.40},
    "Real Estate":            {"ev_sales_median": 6.0,  "peg_median": 2.8, "rev_growth_median": 0.04, "gross_margin_median": 0.50},
    "Communication Services": {"ev_sales_median": 3.0,  "peg_median": 1.8, "rev_growth_median": 0.08, "gross_margin_median": 0.48},
    "Unknown":                {"ev_sales_median": 3.0,  "peg_median": 2.0, "rev_growth_median": 0.07, "gross_margin_median": 0.40},
}


def get_sector_stats(ticker: str) -> Dict[str, Any]:
    """
    Return sector-median valuation benchmarks for relative scoring.
    Looks up the ticker's sector first, then returns the right medians.
    """
    try:
        info   = yf.Ticker(ticker).info or {}
        sector = info.get("sector") or "Unknown"
    except Exception:
        sector = "Unknown"

    return _SECTOR_MEDIANS.get(sector, _SECTOR_MEDIANS["Unknown"])
