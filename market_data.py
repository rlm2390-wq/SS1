# ─────────────────────────────────────────────
#  data/market_data.py  –  Data access layer
#
#  Swap the body of each function for your real
#  data provider (Polygon, Alpaca, Yahoo, etc.)
#  The rest of the bot never changes.
# ─────────────────────────────────────────────
from __future__ import annotations
import datetime
import random
import numpy as np
from typing import Dict, Any, List


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _sim_price_series(n: int = 252, seed: int = 42, trend: float = 0.0003,
                       vol: float = 0.018, start: float = 50.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rets = rng.normal(trend, vol, n)
    prices = start * np.exp(np.cumsum(rets))
    return prices


def _sim_volume_series(n: int = 252, seed: int = 42,
                        base: float = 2_000_000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (base * rng.lognormal(0, 0.4, n)).astype(float)


def _moving_average(series: np.ndarray, window: int) -> np.ndarray:
    ma = np.full_like(series, np.nan)
    for i in range(window - 1, len(series)):
        ma[i] = series[i - window + 1 : i + 1].mean()
    return ma


def _rsi(prices: np.ndarray, period: int = 14) -> float:
    deltas = np.diff(prices[-(period + 1):])
    gains  = deltas[deltas > 0].mean() if (deltas > 0).any() else 0.0
    losses = (-deltas[deltas < 0]).mean() if (deltas < 0).any() else 1e-9
    rs     = gains / (losses + 1e-9)
    return 100 - 100 / (1 + rs)


# ── Public API ────────────────────────────────────────────────────────────────

def get_index_data() -> Dict[str, Any]:
    """
    Return a dict containing index-level data for market-regime scoring.

    Real implementation: pull SPY, QQQ, IWM OHLCV + VIX from your provider.
    """
    rng = np.random.default_rng(0)

    spy = _sim_price_series(252, seed=1, trend=0.0004, vol=0.012, start=450)
    qqq = _sim_price_series(252, seed=2, trend=0.0005, vol=0.015, start=380)
    iwm = _sim_price_series(252, seed=3, trend=0.0002, vol=0.016, start=200)

    vix_base = 18.0
    vix_series = np.clip(vix_base + rng.normal(0, 3, 252), 10, 60)

    # breadth: fraction of 500-stock universe above their 50-day MA (simulated)
    breadth = float(np.clip(rng.normal(0.58, 0.08), 0.1, 0.95))

    return {
        "spy_prices":  spy,
        "qqq_prices":  qqq,
        "iwm_prices":  iwm,
        "spy_ma50":    _moving_average(spy, 50),
        "spy_ma200":   _moving_average(spy, 200),
        "qqq_ma50":    _moving_average(qqq, 50),
        "vix_series":  vix_series,
        "vix_current": float(vix_series[-1]),
        "vix_20d_avg": float(vix_series[-20:].mean()),
        "breadth_pct_above_50ma": breadth,
        "put_call_ratio": float(np.clip(rng.normal(0.85, 0.15), 0.4, 2.0)),
        "advance_decline": float(rng.normal(1.05, 0.15)),  # >1 = more advancers
    }


def get_stock_data(ticker: str, lookback_days: int = 252) -> Dict[str, Any]:
    """
    Return a dict with all per-stock data needed by every brain.

    Real implementation: pull from Polygon/Alpaca/Yahoo and combine.

    Keys returned
    -------------
    prices          np.ndarray  (lookback_days,)
    volumes         np.ndarray  (lookback_days,)
    ma20, ma50, ma200  np.ndarray
    rsi             float
    atr_pct         float  (ATR / last_price)
    bb_width        float  (Bollinger Band width / mid)
    avg_dollar_vol  float
    fundamentals    dict
    short_interest  dict
    insider_activity dict
    news_items      list[dict]
    options_flow    dict
    sector          str
    """
    seed = abs(hash(ticker)) % (2**31)
    rng  = np.random.default_rng(seed)

    # Price / volume
    trend  = rng.uniform(-0.0002, 0.0006)
    vol    = rng.uniform(0.012, 0.040)
    start  = rng.uniform(5.0, 300.0)
    prices  = _sim_price_series(lookback_days, seed=seed, trend=trend, vol=vol, start=start)
    volumes = _sim_volume_series(lookback_days, seed=seed + 1,
                                  base=rng.uniform(500_000, 20_000_000))

    last = float(prices[-1])
    ma20  = _moving_average(prices, 20)
    ma50  = _moving_average(prices, 50)
    ma200 = _moving_average(prices, 200)

    # ATR proxy
    highs = prices * (1 + np.abs(rng.normal(0, 0.008, len(prices))))
    lows  = prices * (1 - np.abs(rng.normal(0, 0.008, len(prices))))
    hl    = highs - lows
    if len(prices) > 1:
        tr = np.maximum(hl[1:], np.maximum(np.abs(highs[1:] - prices[:-1]), np.abs(lows[1:] - prices[:-1])))
    else:
        tr = hl
    atr14 = float(tr[-14:].mean()) if len(tr) >= 14 else float(tr.mean())
    atr_pct = atr14 / last

    # Bollinger Band width
    std20 = float(prices[-20:].std())
    bb_width = (4 * std20) / float(ma20[-1]) if not np.isnan(ma20[-1]) else 0.05

    avg_dollar_vol = float((prices[-20:] * volumes[-20:]).mean())

    # Fundamentals (synthetic)
    sectors = ["Technology", "Healthcare", "Financials", "Consumer", "Energy",
               "Industrials", "Materials", "Utilities", "Real Estate", "Communication"]
    sector = sectors[seed % len(sectors)]

    rev_yoy        = float(rng.normal(0.12, 0.25))
    eps_yoy        = float(rng.normal(0.10, 0.30))
    gross_margin   = float(rng.uniform(0.20, 0.75))
    margin_trend   = float(rng.normal(0.01, 0.05))
    peg            = float(np.clip(rng.normal(1.5, 0.8), 0.1, 8.0))
    ev_sales       = float(np.clip(rng.lognormal(0.8, 0.8), 0.3, 20.0))
    debt_equity    = float(np.clip(rng.exponential(0.6), 0.0, 5.0))
    cash_runway    = float(np.clip(rng.exponential(3.0), 0.2, 15.0))

    fundamentals = {
        "revenue_yoy":       rev_yoy,
        "eps_yoy":           eps_yoy,
        "gross_margin":      gross_margin,
        "margin_trend":      margin_trend,
        "peg":               peg,
        "ev_sales":          ev_sales,
        "debt_to_equity":    debt_equity,
        "cash_runway_years": cash_runway,
        "earnings_surprise": float(rng.normal(0.03, 0.08)),
    }

    # Short interest
    short_float_pct = float(np.clip(rng.exponential(8.0), 0.5, 60.0))
    days_to_cover   = float(np.clip(rng.exponential(3.0), 0.5, 20.0))
    borrow_cost_pct = float(np.clip(rng.exponential(2.0), 0.1, 30.0))

    short_interest = {
        "short_float_pct": short_float_pct,
        "days_to_cover":   days_to_cover,
        "borrow_cost_pct": borrow_cost_pct,
        "short_trend":     float(rng.uniform(-0.3, 0.3)),  # change in short % over 30d
    }

    # Insider activity
    n_buys  = int(rng.poisson(1.0))
    n_sells = int(rng.poisson(1.5))
    net_buy = float(rng.normal(50_000, 300_000)) if n_buys > 0 else 0.0

    insider_activity = {
        "buy_events_90d":     n_buys,
        "sell_events_90d":    n_sells,
        "net_buy_usd_90d":    net_buy,
        "largest_buy_usd":    max(0.0, float(rng.exponential(100_000))),
    }

    # News items  (list of dicts with sentiment scores)
    n_news = int(rng.poisson(3))
    news_items = [
        {
            "headline": f"News item {i+1} for {ticker}",
            "sentiment": float(np.clip(rng.normal(0.05, 0.4), -1, 1)),
            "published_at": (datetime.date.today() - datetime.timedelta(days=int(rng.integers(0, 14)))).isoformat(),
        }
        for i in range(n_news)
    ]

    # Options flow
    call_vol_ratio = float(np.clip(rng.lognormal(0.0, 0.6), 0.2, 8.0))
    call_put_ratio = float(np.clip(rng.lognormal(0.0, 0.4), 0.3, 4.0))

    options_flow = {
        "call_vol_ratio":  call_vol_ratio,   # vs 30d avg
        "call_put_ratio":  call_put_ratio,
        "iv_percentile":   float(rng.uniform(0, 100)),
        "unusual_options": bool(call_vol_ratio > 2.5),
    }

    # Float / ownership
    float_shares_m     = float(np.clip(rng.lognormal(3.5, 1.2), 1.0, 5000.0))
    inst_ownership_pct = float(np.clip(rng.normal(55, 20), 5, 95))

    return {
        "ticker":           ticker,
        "sector":           sector,
        "prices":           prices,
        "volumes":          volumes,
        "ma20":             ma20,
        "ma50":             ma50,
        "ma200":            ma200,
        "rsi":              _rsi(prices),
        "atr_pct":          atr_pct,
        "bb_width":         bb_width,
        "avg_dollar_vol":   avg_dollar_vol,
        "fundamentals":     fundamentals,
        "short_interest":   short_interest,
        "insider_activity": insider_activity,
        "news_items":       news_items,
        "options_flow":     options_flow,
        "float_shares_m":   float_shares_m,
        "inst_ownership_pct": inst_ownership_pct,
        "analyst_upgrades_30d":   int(rng.poisson(0.5)),
        "analyst_downgrades_30d": int(rng.poisson(0.3)),
        "analyst_target_change_pct": float(rng.normal(0.05, 0.15)),
    }


def get_sector_stats(ticker: str) -> Dict[str, Any]:
    """
    Return sector-median valuation stats for relative scoring.

    Real implementation: compute from your universe or pull from a provider.
    """
    seed = abs(hash(ticker + "_sector")) % (2**31)
    rng  = np.random.default_rng(seed)
    return {
        "ev_sales_median":  float(rng.uniform(1.5, 6.0)),
        "peg_median":       float(rng.uniform(1.2, 2.5)),
        "rev_growth_median": float(rng.uniform(0.05, 0.20)),
        "gross_margin_median": float(rng.uniform(0.30, 0.65)),
    }
