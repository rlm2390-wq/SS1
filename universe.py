# ─────────────────────────────────────────────
#  universe.py  –  Dynamic ticker universe
#  Supports multiple universe modes selectable
#  from the dashboard dropdown.
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

from config import (UNIVERSE_CONFIG, FINNHUB_API_KEY,
                    DROP_THRESHOLD, DROP_LOOKBACK_DAYS)

logger = logging.getLogger("universe")

# ── Static fallbacks ──────────────────────────────────────────────────────────

_FALLBACK_TICKERS = [
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","JPM","V","UNH",
    "XOM","MA","JNJ","PG","HD","MRK","COST","ABBV","CVX","CRM",
    "BAC","NFLX","AMD","WMT","KO","PEP","TMO","ADBE","MCD","CSCO",
    "ACN","LIN","ABT","DHR","ORCL","TXN","QCOM","PM","INTU","WFC",
    "CRWD","DDOG","NET","SNOW","PLTR","SOFI","AFRM","IONQ","ACHR","RXRX",
]

_SP500_FALLBACK: list[str] = [
    "AAPL","MSFT","NVDA","GOOGL","GOOG","AMZN","META","BRK-B","LLY","AVGO",
    "TSLA","JPM","V","UNH","XOM","MA","JNJ","PG","HD","MRK","COST","ABBV",
    "CVX","CRM","BAC","NFLX","AMD","WMT","KO","PEP","TMO","ADBE","MCD","CSCO",
    "ACN","LIN","ABT","DHR","ORCL","TXN","QCOM","PM","INTU","WFC","CAT","AMGN",
    "IBM","GE","SPGI","RTX","ISRG","NOW","BKNG","GS","HON","AMAT","VRTX","SYK",
    "BLK","AXP","MDLZ","PLD","ADI","GILD","MMC","ELV","DE","LRCX","MU","REGN",
    "C","PANW","BSX","KLAC","SO","CI","ZTS","CME","DUK","SLB","TJX","AON",
    "SNPS","CDNS","MCO","ITW","EOG","PH","APH","NOC","USB","FI","HCA","EMR",
    "COP","CTAS","MCHP","MSI","ORLY","ADP","CRWD","DDOG","NET","SNOW","PLTR",
    # Extended list
    "F","GM","UBER","LYFT","ABNB","DASH","RBLX","COIN","MSTR","APP",
    "SOFI","AFRM","UPST","HOOD","IONQ","ACHR","JOBY","RKLB","LUNR",
    "MRNA","BNTX","SRPT","RXRX","ARQT","ENPH","FSLR","PLUG","MP",
    "AMT","EQIX","PLD","NEE","SO","TSM","ASML","NVO","SAP","BABA",
    "GME","AMC","NKLA","RIVN","LCID",
]

_SMALL_CAP_TICKERS: list[str] = [
    "IONQ","ACHR","JOBY","RKLB","LUNR","RXRX","ARQT","NKLA","RIVN","LCID",
    "SOFI","AFRM","UPST","HOOD","OPEN","LMND","ROOT","HIMS","CLOV","BARK",
    "SPCE","ASTS","MNTS","SATL","KPLT","PSFE","GENI","DKNG","PENN","EVGO",
    "BLNK","CHPT","WKHS","HYLN","RIDE","GOEV","ELMS","SOLO","IDEX","SHIP",
]

_HIGH_SHORT_TICKERS: list[str] = [
    "GME","AMC","BBBY","NKLA","LCID","RIVN","BYND","OSTK","MVIS","CLOV",
    "WISH","SPCE","IRNT","OPAD","SDC","ATVI","HOOD","COIN","MSTR","RDFN",
    "OPEN","LMND","ROOT","HIMS","BARK","GENI","EVGO","BLNK","CHPT","WKHS",
]


# ── Universe mode definitions ─────────────────────────────────────────────────

UNIVERSE_MODES = {
    "sp500_full":          "S&P 500 Full (~500)",
    "sp500_top100":        "S&P 500 Top 100",
    "small_caps":          "Small Caps",
    "under20":             "Under $20",
    "high_short_interest": "High Short Interest",
    "watchlist":           "My Watchlist",
}


@lru_cache(maxsize=1)
def get_sp500_tickers() -> list[str]:
    if not PD_AVAILABLE:
        logger.warning("pandas unavailable — using fallback S&P 500 list")
        return list(_SP500_FALLBACK)
    try:
        url    = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})
        df     = tables[0]
        tickers = [str(t).replace(".", "-") for t in df["Symbol"].tolist()]
        logger.info("Fetched %d S&P 500 tickers from Wikipedia", len(tickers))
        return tickers
    except Exception as exc:
        logger.error("S&P 500 fetch failed: %s — using fallback", exc)
        return list(_SP500_FALLBACK)


# ── IPO calendar ──────────────────────────────────────────────────────────────

_IPO_CACHE: dict = {}
_IPO_CACHE_TIME: float | None = None
_IPO_CACHE_TTL  = 3600


def get_recent_ipos(days_back: int = 30) -> list[str]:
    global _IPO_CACHE, _IPO_CACHE_TIME
    if _IPO_CACHE_TIME and (time.time() - _IPO_CACHE_TIME) < _IPO_CACHE_TTL:
        return _IPO_CACHE.get("ipos", [])
    if not FINNHUB_API_KEY:
        _IPO_CACHE = {"ipos": []}
        _IPO_CACHE_TIME = time.time()
        return []
    try:
        today     = datetime.utcnow()
        from_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date   = today.strftime("%Y-%m-%d")
        resp = requests.get("https://finnhub.io/api/v1/calendar/ipo",
                            params={"from": from_date, "to": to_date,
                                    "token": FINNHUB_API_KEY}, timeout=5)
        resp.raise_for_status()
        ipos = [e.get("symbol","").upper().strip()
                for e in resp.json().get("ipoCalendar", [])
                if e.get("symbol")]
        _IPO_CACHE = {"ipos": ipos}
        _IPO_CACHE_TIME = time.time()
        return ipos
    except Exception as exc:
        logger.error("IPO fetch failed: %s", exc)
        _IPO_CACHE = {"ipos": []}
        _IPO_CACHE_TIME = time.time()
        return []


# ── Drop scanner ──────────────────────────────────────────────────────────────

_DROP_CACHE: dict = {}
_DROP_CACHE_TIME: float | None = None
_DROP_CACHE_TTL  = 1800


def get_recent_drops(base_tickers: list[str]) -> list[str]:
    global _DROP_CACHE, _DROP_CACHE_TIME
    if _DROP_CACHE_TIME and (time.time() - _DROP_CACHE_TIME) < _DROP_CACHE_TTL:
        return _DROP_CACHE.get("drops", [])
    if not YF_AVAILABLE:
        return []
    drops = []
    for ticker in base_tickers[:50]:
        try:
            hist = yf.Ticker(ticker).history(
                period=f"{DROP_LOOKBACK_DAYS}d", timeout=5)
            time.sleep(0.1)
            if len(hist) >= 2:
                chg = (float(hist["Close"].iloc[-1]) -
                       float(hist["Close"].iloc[0])) / float(hist["Close"].iloc[0])
                if chg < -DROP_THRESHOLD:
                    drops.append(ticker)
        except Exception:
            pass
    _DROP_CACHE = {"drops": drops}
    _DROP_CACHE_TIME = time.time()
    return drops


# ── Public API ────────────────────────────────────────────────────────────────

def get_universe(mode: str | None = None, watchlist: list[str] | None = None) -> list[str]:
    """
    Return ticker list for the given mode.
    mode: "sp500_full" | "sp500_top100" | "small_caps" | "under20" |
          "high_short_interest" | "watchlist"
    """
    # Explicit override
    explicit = UNIVERSE_CONFIG.get("ticker_list")
    if explicit:
        return list(explicit)

    if mode is None:
        mode = UNIVERSE_CONFIG.get("mode", "sp500_top100")

    if mode == "sp500_full":
        sp500 = get_sp500_tickers()
        ipos  = get_recent_ipos()
        drops = get_recent_drops(sp500)
        combined = list(dict.fromkeys(sp500 + ipos + drops))
        return combined or list(_FALLBACK_TICKERS)

    if mode == "sp500_top100":
        return get_sp500_tickers()[:100]

    if mode == "small_caps":
        return list(_SMALL_CAP_TICKERS)

    if mode == "under20":
        # Return small caps + high-short as proxy for sub-$20 names
        return list(dict.fromkeys(_SMALL_CAP_TICKERS + _HIGH_SHORT_TICKERS))

    if mode == "high_short_interest":
        return list(_HIGH_SHORT_TICKERS)

    if mode == "watchlist":
        return list(watchlist or [])

    # Default
    return get_sp500_tickers()[:100]
