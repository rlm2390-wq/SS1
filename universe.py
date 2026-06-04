# ─────────────────────────────────────────────
#  universe.py  –  Dynamic ticker universe
#
#  Builds a 500+ ticker universe from:
#    1. S&P 500 (Wikipedia, cached daily via lru_cache)
#    2. Recent IPOs (Finnhub free tier, 1-hour TTL)
#    3. Recent drops ≥10% in last 5 days (30-min TTL)
#
#  Falls back to a static demo set if all sources fail.
# ─────────────────────────────────────────────
from __future__ import annotations
import logging
import time
from datetime import datetime, timedelta
from functools import lru_cache

import requests

try:
    import pandas as pd
    PD_AVAILABLE = True
except ImportError:
    PD_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

from config import UNIVERSE_CONFIG, FINNHUB_API_KEY, DROP_THRESHOLD, DROP_LOOKBACK_DAYS

logger = logging.getLogger("universe")

# ── Static fallback (used if all dynamic sources fail) ────────────────────────
_FALLBACK_TICKERS = [
    # Large cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
    # Mid cap growth
    "CRWD", "SNOW", "DDOG", "NET", "BILL", "GTLB",
    # Small cap
    "IONQ", "ARQT", "ACHR", "JOBY", "RXRX",
    # Healthcare
    "MRNA", "BNTX", "SRPT",
    # Financials
    "SOFI", "AFRM", "UPST",
    # Energy
    "ENPH", "PLUG",
    # Classic value / squeeze candidates
    "GME", "AMC", "BBBY",
]


# ── S&P 500 list (cached for the lifetime of the process via lru_cache) ───────

# Hardcoded top-100 S&P 500 tickers used as a fallback when Wikipedia/lxml
# is unavailable.  Ordered roughly by market cap (as of mid-2024).
_SP500_FALLBACK: list[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "BRK-B",
    "LLY", "AVGO", "TSLA", "JPM", "V", "UNH", "XOM", "MA", "JNJ", "PG",
    "HD", "MRK", "COST", "ABBV", "CVX", "CRM", "BAC", "NFLX", "AMD",
    "WMT", "KO", "PEP", "TMO", "ADBE", "MCD", "CSCO", "ACN", "LIN",
    "ABT", "DHR", "ORCL", "TXN", "QCOM", "PM", "INTU", "WFC", "CAT",
    "AMGN", "IBM", "GE", "SPGI", "RTX", "ISRG", "NOW", "BKNG", "GS",
    "HON", "AMAT", "VRTX", "SYK", "BLK", "AXP", "MDLZ", "PLD", "ADI",
    "GILD", "MMC", "ELV", "DE", "LRCX", "MU", "REGN", "C", "PANW",
    "BSX", "KLAC", "SO", "CI", "ZTS", "CME", "DUK", "SLB", "TJX",
    "AON", "SNPS", "CDNS", "MCO", "ITW", "EOG", "PH", "APH", "NOC",
    "USB", "FI", "HCA", "EMR", "COP", "CTAS", "MCHP", "MSI", "ORLY",
    "ADP", "CRWD", "DDOG", "NET", "SNOW", "PLTR",
]


@lru_cache(maxsize=1)
def get_sp500_tickers() -> list[str]:
    """
    Fetch S&P 500 tickers from Wikipedia (requires lxml).
    Falls back to a hardcoded top-100 list if the fetch fails.
    Cached for the process lifetime (effectively daily since the service
    restarts daily on Railway).
    """
    if not PD_AVAILABLE:
        logger.warning(
            "pandas not available — cannot fetch S&P 500 list; "
            "using hardcoded top-100 fallback (%d tickers)",
            len(_SP500_FALLBACK),
        )
        return list(_SP500_FALLBACK)
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})
        df = tables[0]
        tickers = df["Symbol"].tolist()
        # Wikipedia uses dots for BRK.B etc.; yfinance expects hyphens
        tickers = [str(t).replace(".", "-") for t in tickers]
        logger.info("Fetched %d S&P 500 tickers from Wikipedia", len(tickers))
        return tickers
    except Exception as exc:
        logger.error(
            "Failed to fetch S&P 500 list from Wikipedia (%s); "
            "using hardcoded top-100 fallback (%d tickers)",
            exc, len(_SP500_FALLBACK),
        )
        return list(_SP500_FALLBACK)


# ── IPO calendar (Finnhub free tier, 1-hour TTL) ──────────────────────────────

_IPO_CACHE: dict = {}
_IPO_CACHE_TIME: float | None = None
_IPO_CACHE_TTL = 3600  # 1 hour


def get_recent_ipos(days_back: int = 30) -> list[str]:
    """
    Fetch tickers that IPO'd in the last *days_back* days via Finnhub.
    Requires FINNHUB_API_KEY env var; returns [] if not set.
    Results are cached for 1 hour.
    """
    global _IPO_CACHE, _IPO_CACHE_TIME

    if _IPO_CACHE_TIME and (time.time() - _IPO_CACHE_TIME) < _IPO_CACHE_TTL:
        logger.debug("get_recent_ipos: returning cached IPO list (%d tickers)",
                     len(_IPO_CACHE.get("ipos", [])))
        return _IPO_CACHE.get("ipos", [])

    if not FINNHUB_API_KEY:
        logger.warning("FINNHUB_API_KEY not set — IPO detection disabled")
        _IPO_CACHE = {"ipos": []}
        _IPO_CACHE_TIME = time.time()
        return []

    try:
        today = datetime.utcnow()
        from_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")

        url = "https://finnhub.io/api/v1/calendar/ipo"
        params = {"from": from_date, "to": to_date, "token": FINNHUB_API_KEY}
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        ipos: list[str] = []
        for event in data.get("ipoCalendar", []):
            ticker = event.get("symbol", "").upper().strip()
            if ticker:
                ipos.append(ticker)

        _IPO_CACHE = {"ipos": ipos}
        _IPO_CACHE_TIME = time.time()
        logger.info("Fetched %d recent IPOs from Finnhub", len(ipos))
        return ipos

    except Exception as exc:
        logger.error("Failed to fetch IPO calendar: %s", exc)
        _IPO_CACHE = {"ipos": []}
        _IPO_CACHE_TIME = time.time()
        return []


# ── Drop scanner (tickers down ≥10% in last 1-5 days, 30-min TTL) ─────────────

_DROP_CACHE: dict = {}
_DROP_CACHE_TIME: float | None = None
_DROP_CACHE_TTL = 1800  # 30 minutes


def get_recent_drops(
    base_tickers: list[str],
    drop_threshold: float = DROP_THRESHOLD,
    lookback_days: int = DROP_LOOKBACK_DAYS,
) -> list[str]:
    """
    Scan the first 50 tickers in *base_tickers* for those that have fallen
    *drop_threshold* (e.g. 0.10 = 10%) or more over the last *lookback_days*.
    Results are cached for 30 minutes.
    """
    global _DROP_CACHE, _DROP_CACHE_TIME

    if _DROP_CACHE_TIME and (time.time() - _DROP_CACHE_TIME) < _DROP_CACHE_TTL:
        logger.debug("get_recent_drops: returning cached drop list (%d tickers)",
                     len(_DROP_CACHE.get("drops", [])))
        return _DROP_CACHE.get("drops", [])

    if not YF_AVAILABLE:
        logger.warning("yfinance not available — drop detection disabled")
        return []

    drops: list[dict] = []

    for ticker in base_tickers[:50]:  # cap at 50 to respect rate limits
        try:
            hist = yf.Ticker(ticker).history(period=f"{lookback_days}d", timeout=5)
            time.sleep(0.1)
            if len(hist) >= 2:
                open_price = float(hist["Close"].iloc[0])
                close_price = float(hist["Close"].iloc[-1])
                if open_price > 0:
                    change = (close_price - open_price) / open_price
                    if change < -drop_threshold:
                        drops.append({"ticker": ticker, "change": change})
                        logger.debug("Drop detected: %s down %.1f%%",
                                     ticker, change * 100)
        except Exception as exc:
            logger.debug("Drop scan skipped for %s: %s", ticker, exc)

    drops.sort(key=lambda x: x["change"])  # most dropped first
    drop_tickers = [d["ticker"] for d in drops]

    _DROP_CACHE = {"drops": drop_tickers}
    _DROP_CACHE_TIME = time.time()
    logger.info(
        "Drop scan complete: %d tickers down >%.0f%% in last %d days",
        len(drop_tickers), drop_threshold * 100, lookback_days,
    )
    return drop_tickers


# ── Smart universe builder ─────────────────────────────────────────────────────

def get_universe() -> list[str]:
    """
    Build the dynamic scan universe:
      1. Honour explicit UNIVERSE_CONFIG["ticker_list"] if set.
      2. Otherwise: S&P 500 + recent IPOs + recent drops (deduplicated).
      3. Fall back to the static demo set if all dynamic sources return empty.
    """
    explicit = UNIVERSE_CONFIG.get("ticker_list")
    if explicit:
        logger.info("get_universe: using explicit ticker_list (%d tickers)", len(explicit))
        return list(explicit)

    sp500 = get_sp500_tickers()
    ipos  = get_recent_ipos(days_back=30)
    drops = get_recent_drops(sp500, drop_threshold=DROP_THRESHOLD,
                             lookback_days=DROP_LOOKBACK_DAYS)

    combined = list(dict.fromkeys(sp500 + ipos + drops))  # dedup, preserve order

    if not combined:
        logger.warning("All dynamic sources returned empty — falling back to demo tickers")
        return list(_FALLBACK_TICKERS)

    logger.info(
        "Universe built: %d S&P 500 + %d IPOs + %d drops = %d unique tickers",
        len(sp500), len(ipos), len(drops), len(combined),
    )
    return combined
