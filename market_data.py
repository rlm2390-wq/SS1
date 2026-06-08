# ─────────────────────────────────────────────
#  market_data.py  –  Real data via yfinance
#
#  Rate-limit safe: sequential fetching with
#  exponential backoff and per-call delays.
# ─────────────────────────────────────────────
from __future__ import annotations
import datetime
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

import cache as _cache
from config import (REQUEST_DELAY_SECONDS, REQUEST_MAX_RETRIES,
                    REQUEST_RETRY_BACKOFF)

logger = logging.getLogger("market_data")

# ── Request helpers ───────────────────────────────────────────────────────────

def _yf_fetch(fn, *args, retries=REQUEST_MAX_RETRIES, **kwargs):
    """Call a yfinance function with retry + backoff on rate limit errors."""
    delay = REQUEST_DELAY_SECONDS
    for attempt in range(retries):
        try:
            time.sleep(delay)
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            if "too many requests" in msg or "rate limit" in msg or "429" in msg:
                wait = delay * (REQUEST_RETRY_BACKOFF ** attempt)
                logger.warning("Rate limited on attempt %d/%d — sleeping %.1fs",
                               attempt + 1, retries, wait)
                time.sleep(wait)
                delay = wait
            else:
                raise
    raise RuntimeError(f"Exceeded {retries} retries")


# ── Math helpers ──────────────────────────────────────────────────────────────

def _safe(val, default=0.0):
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
        ma[i] = series[i - window + 1: i + 1].mean()
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
    sl  = prices[-window:]
    mid = sl.mean()
    std = sl.std()
    return float(4 * std / (mid + 1e-9))


# ── Index / market data ───────────────────────────────────────────────────────

def get_index_data() -> Dict[str, Any]:
    if not YF_AVAILABLE:
        raise RuntimeError("yfinance not installed")

    def _fetch(ticker):
        t    = yf.Ticker(ticker)
        hist = _yf_fetch(t.history, period="1y")
        return hist["Close"].values.astype(float)

    spy = _fetch("SPY")
    qqq = _fetch("QQQ")
    iwm = _fetch("IWM")
    vix = _fetch("^VIX")

    # Breadth proxy — sample 15 Dow stocks
    dow_sample = ["AAPL","MSFT","JPM","JNJ","V","WMT","PG","UNH","HD",
                  "CVX","MRK","AMGN","CAT","GS","MMM"]
    above_50 = 0
    for sym in dow_sample:
        try:
            h  = _yf_fetch(yf.Ticker(sym).history, period="3mo")["Close"].values
            ma = _moving_average(h, 50)
            if not np.isnan(ma[-1]) and h[-1] > ma[-1]:
                above_50 += 1
        except Exception:
            pass
    breadth = above_50 / len(dow_sample)

    vix_cur = float(vix[-1])   if len(vix) else 20.0
    vix_avg = float(vix[-20:].mean()) if len(vix) >= 20 else vix_cur
    pc_proxy = float(np.clip(vix_cur / 20.0 * 0.85, 0.4, 2.0))

    return {
        "spy_prices":             spy,
        "qqq_prices":             qqq,
        "iwm_prices":             iwm,
        "spy_ma50":               _moving_average(spy, 50),
        "spy_ma200":              _moving_average(spy, 200),
        "qqq_ma50":               _moving_average(qqq, 50),
        "spy_last":               float(spy[-1]),
        "qqq_last":               float(qqq[-1]),
        "iwm_last":               float(iwm[-1]),
        "spy_pct_chg":            float((spy[-1] - spy[-2]) / spy[-2]) if len(spy) > 1 else 0.0,
        "vix_series":             vix,
        "vix_current":            vix_cur,
        "vix_20d_avg":            vix_avg,
        "breadth_pct_above_50ma": breadth,
        "put_call_ratio":         pc_proxy,
        "advance_decline":        1.0,
    }


# ── Stock data ────────────────────────────────────────────────────────────────

def get_stock_data(ticker: str, lookback_days: int = 252) -> Dict[str, Any]:
    # Check cache first — daily data is valid for 10 minutes
    cache_key = f"stock_data:{ticker}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    if not YF_AVAILABLE:
        raise RuntimeError("yfinance not installed")

    try:
        t    = yf.Ticker(ticker)
        hist = _yf_fetch(t.history, period="1y")
        info = t.info or {}
    except Exception as e:
        logger.warning("get_stock_data: yfinance fetch failed for %s: %s — skipping", ticker, e)
        return {}

    # If 1y returns empty (can happen on weekends/holidays), try shorter period
    if hist.empty:
        try:
            hist = _yf_fetch(t.history, period="6mo")
        except Exception:
            return {}
    if hist.empty or len(hist) < 10:
        return {}

    prices  = hist["Close"].values.astype(float)
    volumes = hist["Volume"].values.astype(float)
    hi      = hist["High"].values.astype(float)
    lo      = hist["Low"].values.astype(float)

    if len(prices) > lookback_days:
        prices  = prices[-lookback_days:]
        volumes = volumes[-lookback_days:]
        hi      = hi[-lookback_days:]
        lo      = lo[-lookback_days:]

    last  = float(prices[-1])
    ma20  = _moving_average(prices, 20)
    ma50  = _moving_average(prices, 50)
    ma200 = _moving_average(prices, 200)

    if len(hi) > 1:
        tr    = np.maximum(hi[1:] - lo[1:],
                np.maximum(np.abs(hi[1:] - prices[:-1]),
                           np.abs(lo[1:] - prices[:-1])))
        atr14 = float(tr[-14:].mean()) if len(tr) >= 14 else float(tr.mean())
    else:
        atr14 = last * 0.02
    atr_pct = atr14 / (last + 1e-9)

    avg_dollar_vol = float((prices[-20:] * volumes[-20:]).mean())

    # Pre/after market
    pre_market  = _safe(info.get("preMarketPrice"),  last)
    post_market = _safe(info.get("postMarketPrice"), last)
    pre_chg     = (pre_market  - last) / (last + 1e-9) if pre_market  != last else 0.0
    post_chg    = (post_market - last) / (last + 1e-9) if post_market != last else 0.0

    # Beta vs SPY
    beta = _safe(info.get("beta"), 1.0)

    # Fundamentals
    fundamentals = {
        "revenue_yoy":       _safe(info.get("revenueGrowth"),          0.05),
        "eps_yoy":           _safe(info.get("earningsGrowth"),         0.05),
        "gross_margin":      _safe(info.get("grossMargins"),           0.40),
        "margin_trend":      _safe(info.get("operatingMargins"), 0.10) - 0.10,
        "peg":               _safe(info.get("pegRatio"),               2.0),
        "ev_sales":          _safe(info.get("enterpriseToRevenue"),    3.0),
        "debt_to_equity":    _safe(info.get("debtToEquity"), 50.0) / 100.0,
        "cash_runway_years": _safe(info.get("totalCash"), 0) /
                             max(_safe(info.get("totalRevenue"), 1), 1) * 2,
        "earnings_surprise": _safe(info.get("earningsQuarterlyGrowth"), 0.0),
        "market_cap":        _safe(info.get("marketCap"), 0),
    }

    # Short interest
    shares_short   = _safe(info.get("sharesShort"), 0)
    avg_vol_10     = _safe(info.get("averageVolume10days"),
                           volumes[-10:].mean() if len(volumes) >= 10 else volumes.mean())
    short_float_pct = _safe(info.get("shortPercentOfFloat"), 0.05) * 100
    days_to_cover   = shares_short / max(avg_vol_10, 1)

    short_interest = {
        "short_float_pct": short_float_pct,
        "days_to_cover":   float(np.clip(days_to_cover, 0, 30)),
        "borrow_cost_pct": float(np.clip(short_float_pct * 0.3, 0.1, 30.0)),
        "short_trend":     0.0,
    }

    # Insider activity
    try:
        ins_df = t.insider_purchases
        if ins_df is not None and not ins_df.empty and "Shares" in ins_df.columns:
            buys  = len(ins_df[ins_df.get("Transaction",
                        ins_df.columns[0]).str.contains("Buy|Purchase", na=False)])
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

    # News
    news_items = []
    try:
        raw_news = t.news or []
        pos_words = ["beat","surge","soar","growth","record","upgrade","raise",
                     "rally","strong","profit","bullish","buy","gain","rise"]
        neg_words = ["miss","fall","drop","cut","downgrade","loss","warn",
                     "decline","weak","bearish","sell","risk","plunge","crash"]
        for n in raw_news[:10]:
            title = n.get("title", "")
            score = (sum(1 for w in pos_words if w in title.lower()) -
                     sum(1 for w in neg_words if w in title.lower()))
            sentiment = float(np.clip(score / 3.0, -1.0, 1.0))
            pub = n.get("providerPublishTime", 0)
            pub_date = (datetime.datetime.utcfromtimestamp(pub).strftime("%Y-%m-%d")
                        if pub else datetime.date.today().isoformat())
            news_items.append({
                "headline":    title,
                "sentiment":   sentiment,
                "published_at": pub_date,
                "url":         n.get("link", ""),
                "source":      n.get("publisher", ""),
            })
    except Exception:
        pass

    # Options flow
    options_flow = {
        "call_vol_ratio": 1.0,
        "call_put_ratio": 1.0,
        "iv_percentile":  50.0,
        "unusual_options": False,
        "iv_rank":        50.0,
    }
    try:
        exp_dates = t.options
        if exp_dates:
            chain     = t.option_chain(exp_dates[0])
            call_vol  = float(chain.calls["volume"].sum())
            put_vol   = float(chain.puts["volume"].sum())
            cpr       = call_vol / max(put_vol, 1)
            avg_iv    = float(chain.calls["impliedVolatility"].mean() * 100)
            options_flow = {
                "call_vol_ratio":  float(np.clip(cpr, 0.1, 5.0)),
                "call_put_ratio":  float(np.clip(cpr, 0.1, 5.0)),
                "iv_percentile":   float(np.clip(avg_iv, 0, 100)),
                "iv_rank":         float(np.clip(avg_iv, 0, 100)),
                "unusual_options": bool(cpr > 2.5),
            }
    except Exception:
        pass

    # Earnings date
    earnings_date = None
    try:
        cal = t.calendar
        if cal is not None and not cal.empty:
            ed = cal.get("Earnings Date")
            if ed is not None and len(ed) > 0:
                earnings_date = str(ed.iloc[0])[:10]
    except Exception:
        pass

    result = {
        "ticker":              ticker,
        "sector":              info.get("sector") or info.get("industry") or "Unknown",
        "industry":            info.get("industry", ""),
        "prices":              prices,
        "volumes":             volumes,
        "ma20":                ma20,
        "ma50":                ma50,
        "ma200":               ma200,
        "rsi":                 _rsi(prices),
        "atr_pct":             atr_pct,
        "bb_width":            _bb_width(prices),
        "avg_dollar_vol":      avg_dollar_vol,
        "fundamentals":        fundamentals,
        "short_interest":      short_interest,
        "insider_activity":    insider_activity,
        "news_items":          news_items,
        "options_flow":        options_flow,
        "float_shares_m":      _safe(info.get("floatShares"), 500e6) / 1e6,
        "inst_ownership_pct":  _safe(info.get("heldPercentInstitutions"), 0.5) * 100,
        "analyst_upgrades_30d":      0,
        "analyst_downgrades_30d":    0,
        "analyst_target_change_pct": (
            _safe(info.get("targetMeanPrice"), last) / (last + 1e-9) - 1.0),
        "analyst_target_price":      _safe(info.get("targetMeanPrice"), 0),
        "beta":                beta,
        "pre_market_price":    pre_market,
        "post_market_price":   post_market,
        "pre_market_chg":      pre_chg,
        "post_market_chg":     post_chg,
        "earnings_date":       earnings_date,
        "52w_high":            float(prices[-252:].max()) if len(prices) >= 252 else float(prices.max()),
        "52w_low":             float(prices[-252:].min()) if len(prices) >= 252 else float(prices.min()),
        "market_cap":          _safe(info.get("marketCap"), 0),
    }

    _cache.set(f"stock_data:{ticker}", result, _cache.TTL_DAILY)
    return result


# ── Sector stats ──────────────────────────────────────────────────────────────

_SECTOR_MEDIANS = {
    "Technology":             {"ev_sales_median": 6.0,  "peg_median": 2.0,  "rev_growth_median": 0.12, "gross_margin_median": 0.62},
    "Healthcare":             {"ev_sales_median": 3.5,  "peg_median": 2.2,  "rev_growth_median": 0.08, "gross_margin_median": 0.55},
    "Financials":             {"ev_sales_median": 2.5,  "peg_median": 1.5,  "rev_growth_median": 0.07, "gross_margin_median": 0.45},
    "Consumer Cyclical":      {"ev_sales_median": 1.8,  "peg_median": 1.8,  "rev_growth_median": 0.06, "gross_margin_median": 0.38},
    "Consumer Defensive":     {"ev_sales_median": 1.5,  "peg_median": 2.0,  "rev_growth_median": 0.04, "gross_margin_median": 0.35},
    "Energy":                 {"ev_sales_median": 1.2,  "peg_median": 1.2,  "rev_growth_median": 0.05, "gross_margin_median": 0.28},
    "Industrials":            {"ev_sales_median": 2.0,  "peg_median": 1.8,  "rev_growth_median": 0.06, "gross_margin_median": 0.33},
    "Materials":              {"ev_sales_median": 1.8,  "peg_median": 1.5,  "rev_growth_median": 0.05, "gross_margin_median": 0.30},
    "Utilities":              {"ev_sales_median": 2.2,  "peg_median": 2.5,  "rev_growth_median": 0.03, "gross_margin_median": 0.40},
    "Real Estate":            {"ev_sales_median": 6.0,  "peg_median": 2.8,  "rev_growth_median": 0.04, "gross_margin_median": 0.50},
    "Communication Services": {"ev_sales_median": 3.0,  "peg_median": 1.8,  "rev_growth_median": 0.08, "gross_margin_median": 0.48},
    "Unknown":                {"ev_sales_median": 3.0,  "peg_median": 2.0,  "rev_growth_median": 0.07, "gross_margin_median": 0.40},
}


def get_sector_stats(ticker: str) -> Dict[str, Any]:
    sector = "Unknown"
    try:
        info   = yf.Ticker(ticker).info or {}
        sector = info.get("sector") or "Unknown"
    except Exception as e:
        logger.warning("get_sector_stats: failed to fetch sector for %s: %s", ticker, e)
    return _SECTOR_MEDIANS.get(sector, _SECTOR_MEDIANS["Unknown"])


# ── Ticker detail (for drawer) ────────────────────────────────────────────────

def get_ticker_detail(ticker: str) -> Dict[str, Any]:
    """
    Full detail payload for the ticker drawer.
    Includes news, peers, earnings, pre/post market, beta.
    """
    try:
        t    = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as e:
        logger.warning("get_ticker_detail failed for %s: %s", ticker, e)
        return {"ticker": ticker, "error": str(e)}

    sector = info.get("sector", "Unknown")

    # Peers — find sector peers from S&P 500 fallback list
    try:
        from universe import get_sp500_tickers
        sp500 = get_sp500_tickers()
        peers_raw = []
        for pt in sp500[:50]:
            if pt == ticker:
                continue
            try:
                pi = yf.Ticker(pt).info or {}
                if pi.get("sector") == sector:
                    peers_raw.append({
                        "ticker":     pt,
                        "price":      _safe(pi.get("currentPrice") or pi.get("regularMarketPrice"), 0),
                        "market_cap": _safe(pi.get("marketCap"), 0),
                    })
            except Exception:
                pass
            if len(peers_raw) >= 4:
                break
    except Exception:
        peers_raw = []

    # Earnings calendar
    earnings_date = None
    earnings_estimate = None
    try:
        cal = t.calendar
        if cal is not None and not cal.empty:
            ed = cal.get("Earnings Date")
            ee = cal.get("EPS Estimate")
            if ed is not None and len(ed) > 0:
                earnings_date = str(ed.iloc[0])[:10]
            if ee is not None and len(ee) > 0:
                earnings_estimate = _safe(ee.iloc[0], None)
    except Exception:
        pass

    # News
    news = []
    try:
        raw  = t.news or []
        pos_words = ["beat","surge","soar","growth","record","upgrade","raise","rally","strong","profit"]
        neg_words = ["miss","fall","drop","cut","downgrade","loss","warn","decline","weak","bearish"]
        for n in raw[:10]:
            title = n.get("title", "")
            score = (sum(1 for w in pos_words if w in title.lower()) -
                     sum(1 for w in neg_words if w in title.lower()))
            sentiment = float(np.clip(score / 3.0, -1.0, 1.0))
            pub = n.get("providerPublishTime", 0)
            pub_date = (datetime.datetime.utcfromtimestamp(pub).strftime("%Y-%m-%d")
                        if pub else "")
            news.append({
                "headline":    title,
                "sentiment":   sentiment,
                "published_at": pub_date,
                "url":         n.get("link", ""),
                "source":      n.get("publisher", ""),
            })
    except Exception:
        pass

    last = _safe(info.get("currentPrice") or info.get("regularMarketPrice"), 0)
    pre  = _safe(info.get("preMarketPrice"),  last)
    post = _safe(info.get("postMarketPrice"), last)

    return {
        "ticker":           ticker,
        "company_name":     info.get("longName", ticker),
        "sector":           sector,
        "industry":         info.get("industry", ""),
        "description":      info.get("longBusinessSummary", "")[:400],
        "price":            last,
        "pre_market_price": pre,
        "post_market_price": post,
        "pre_market_chg":   (pre  - last) / (last + 1e-9),
        "post_market_chg":  (post - last) / (last + 1e-9),
        "52w_high":         _safe(info.get("fiftyTwoWeekHigh"), 0),
        "52w_low":          _safe(info.get("fiftyTwoWeekLow"),  0),
        "market_cap":       _safe(info.get("marketCap"),        0),
        "beta":             _safe(info.get("beta"),             1.0),
        "pe_ratio":         _safe(info.get("trailingPE"),       0),
        "analyst_target":   _safe(info.get("targetMeanPrice"),  0),
        "analyst_rec":      info.get("recommendationKey", ""),
        "earnings_date":    earnings_date,
        "earnings_estimate": earnings_estimate,
        "news":             news,
        "peers":            peers_raw,
    }


# ── Earnings calendar ─────────────────────────────────────────────────────────

def get_earnings_calendar(tickers: List[str], days_ahead: int = 14) -> List[Dict]:
    """Return tickers in list with earnings in next days_ahead days."""
    results = []
    today   = datetime.date.today()
    cutoff  = today + datetime.timedelta(days=days_ahead)

    for ticker in tickers:
        try:
            t   = yf.Ticker(ticker)
            cal = t.calendar
            if cal is None or cal.empty:
                continue
            ed = cal.get("Earnings Date")
            if ed is None or len(ed) == 0:
                continue
            ed_str = str(ed.iloc[0])[:10]
            ed_date = datetime.date.fromisoformat(ed_str)
            if today <= ed_date <= cutoff:
                info = t.info or {}
                # Expected move from IV
                iv = 0.0
                try:
                    exps = t.options
                    if exps:
                        chain = t.option_chain(exps[0])
                        iv = float(chain.calls["impliedVolatility"].mean() * 100)
                except Exception:
                    pass
                days_away = (ed_date - today).days
                results.append({
                    "ticker":       ticker,
                    "earnings_date": ed_str,
                    "days_away":    days_away,
                    "expected_move": round(iv / 100 * 0.8, 3),   # rough: IV → expected move
                    "sector":       info.get("sector", "Unknown"),
                    "price":        _safe(info.get("currentPrice") or
                                         info.get("regularMarketPrice"), 0),
                })
        except Exception:
            pass

    results.sort(key=lambda x: x["days_away"])
    return results
