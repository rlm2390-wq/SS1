# ─────────────────────────────────────────────
#  scheduler.py  –  Section-based async scheduler
#
#  Fetches Webull real-time data at per-section
#  intervals. Updates shared cache. Does NOT
#  run scoring (that stays in yfinance scan loop).
#
#  Section cadences (API fetch):
#    top_alerts / low_float / vol_anomaly /
#    under5 / penny_radar    → 5s
#    under10 / short_squeeze → 10s
#    squeeze_box             → 15s
#    pre_earnings            → 30s
# ─────────────────────────────────────────────
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from config import REFRESH_INTERVALS, WEBULL_ENABLED
from data_webull import WebullClient, get_client

logger = logging.getLogger("scheduler")

# ── Shared real-time cache ────────────────────────────────────────────────────
# Written by scheduler, read by dashboard_backend and app.py

_rt_cache: Dict[str, Any] = {
    "quotes":         {},   # {ticker: quote_dict}
    "intraday":       {},   # {ticker: [candle, ...]}
    "options":        {},   # {ticker: options_chain}
    "last_updated":   {},   # {section: timestamp}
    "section_quotes": {},   # {section: {ticker: quote}}
}
_cache_lock = asyncio.Lock()


def get_rt_cache() -> Dict[str, Any]:
    """Read-only access to the real-time cache."""
    return _rt_cache


def get_quote_cached(ticker: str) -> Optional[Dict]:
    return _rt_cache["quotes"].get(ticker)


def get_section_quotes(section: str) -> Dict[str, Dict]:
    return _rt_cache["section_quotes"].get(section, {})


# ── Section ticker resolvers ──────────────────────────────────────────────────
# These pull ticker lists from the main scan results.
# Import is deferred to avoid circular imports.

def _get_tickers_from_results(section_key: str, max_n: int = 50) -> List[str]:
    """Pull tickers relevant to a section from the last scan results."""
    try:
        from app import _last_results, _last_alerts  # type: ignore
        results = list(_last_results)
    except ImportError:
        return []

    if section_key == "top_alerts":
        try:
            from app import _last_alerts as _a  # type: ignore
            return [r["ticker"] for r in _a[:max_n]]
        except ImportError:
            return []

    if section_key == "under_10":
        return [r["ticker"] for r in results if r.get("last_price", 999) <= 10][:max_n]

    if section_key == "under_5":
        return [r["ticker"] for r in results if r.get("last_price", 999) <= 5][:max_n]

    if section_key in ("low_float_rockets",):
        # Low float: float_ownership sub-score proxy
        lf = [r for r in results
              if r.get("sub_scores", {}).get("structural", {}).get("float_ownership", 1) < 0.35]
        return [r["ticker"] for r in lf[:max_n]]

    if section_key == "short_squeeze":
        sq = [r for r in results if "short_squeeze" in (r.get("setups") or [])]
        return [r["ticker"] for r in sq[:max_n]]

    if section_key == "volume_anomaly":
        return [r["ticker"] for r in results
                if r.get("microstructure", {}).get("sub_scores", {}).get("dollar_vol_accel", 0) >= 0.6
               ][:max_n]

    if section_key == "penny_radar":
        return [r["ticker"] for r in results if 0.50 <= r.get("last_price", 999) <= 2.0][:max_n]

    if section_key == "pre_earnings":
        return [r["ticker"] for r in results if r.get("earnings_date")][:max_n]

    if section_key == "squeeze_box":
        return [r["ticker"] for r in results
                if any(s["name"] == "vol_contraction"
                       for s in r.get("pre_signals", {}).get("signals", []))
               ][:max_n]

    # Default: top results by upside
    return [r["ticker"] for r in results[:max_n]]


# ── Section fetch tasks ───────────────────────────────────────────────────────

async def _fetch_section(
    section_key: str,
    webull: WebullClient,
    include_candles: bool = False,
    include_options: bool = False,
) -> None:
    """Fetch Webull real-time data for a section and update cache."""
    tickers = await asyncio.to_thread(_get_tickers_from_results, section_key)

    if not tickers:
        logger.debug("Section %s: no tickers", section_key)
        return

    # Batch quotes
    quotes = await asyncio.to_thread(webull.get_quotes_batch, tickers)

    async with _cache_lock:
        _rt_cache["quotes"].update(quotes)
        _rt_cache["section_quotes"][section_key] = quotes
        _rt_cache["last_updated"][section_key]   = time.time()

    # Optional intraday candles (slower)
    if include_candles:
        for ticker in tickers[:10]:   # limit candle fetches
            candles = await asyncio.to_thread(
                webull.get_intraday_candles, ticker, "m1", 60
            )
            async with _cache_lock:
                _rt_cache["intraday"][ticker] = candles

    logger.debug("Section %s updated: %d quotes", section_key, len(quotes))


# ── Individual section coroutines ─────────────────────────────────────────────

async def run_section_top_alerts(webull: WebullClient) -> None:
    await _fetch_section("top_alerts", webull, include_candles=True)

async def run_section_under_10(webull: WebullClient) -> None:
    await _fetch_section("under_10", webull)

async def run_section_under_5(webull: WebullClient) -> None:
    await _fetch_section("under_5", webull)

async def run_section_low_float_rockets(webull: WebullClient) -> None:
    await _fetch_section("low_float_rockets", webull)

async def run_section_short_squeeze(webull: WebullClient) -> None:
    await _fetch_section("short_squeeze", webull)

async def run_section_volume_anomalies(webull: WebullClient) -> None:
    await _fetch_section("volume_anomaly", webull)

async def run_section_squeeze_box(webull: WebullClient) -> None:
    await _fetch_section("squeeze_box", webull)

async def run_section_pre_earnings(webull: WebullClient) -> None:
    await _fetch_section("pre_earnings", webull, include_options=True)

async def run_section_penny_radar(webull: WebullClient) -> None:
    await _fetch_section("penny_radar", webull)


# ── Periodic task wrapper ─────────────────────────────────────────────────────

async def periodic_task(
    fn:       Callable,
    interval: int,
    webull:   WebullClient,
) -> None:
    """Run fn(webull) every `interval` seconds indefinitely."""
    while True:
        try:
            await fn(webull)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Section task %s error: %s", fn.__name__, e)
        await asyncio.sleep(interval)


# ── Main scheduler entry point ────────────────────────────────────────────────

async def scheduler_main() -> None:
    """
    Launch all section tasks concurrently.
    Call this from a thread: asyncio.run(scheduler_main())
    """
    webull = get_client()

    if not webull.is_available():
        logger.warning(
            "Webull credentials not set — real-time scheduler idle. "
            "Set WEBULL_APP_KEY + WEBULL_APP_SECRET to enable."
        )
        # Keep running but do nothing — graceful degradation
        while True:
            await asyncio.sleep(60)

    logger.info("Webull scheduler starting — all sections active")

    tasks = [
        periodic_task(run_section_top_alerts,       REFRESH_INTERVALS["top_alerts"],        webull),
        periodic_task(run_section_under_10,          REFRESH_INTERVALS["under_10"],          webull),
        periodic_task(run_section_under_5,           REFRESH_INTERVALS["under_5"],           webull),
        periodic_task(run_section_low_float_rockets, REFRESH_INTERVALS["low_float_rockets"], webull),
        periodic_task(run_section_short_squeeze,     REFRESH_INTERVALS["short_squeeze"],     webull),
        periodic_task(run_section_volume_anomalies,  REFRESH_INTERVALS["volume_anomalies"],  webull),
        periodic_task(run_section_squeeze_box,       REFRESH_INTERVALS["squeeze_box"],       webull),
        periodic_task(run_section_pre_earnings,      REFRESH_INTERVALS["pre_earnings"],      webull),
        periodic_task(run_section_penny_radar,       REFRESH_INTERVALS["penny_radar"],       webull),
    ]

    await asyncio.gather(*tasks)


def start_scheduler_thread() -> None:
    """
    Launch the async scheduler in a background daemon thread.
    Call once at app startup alongside the yfinance scan thread.
    """
    import threading

    def _run():
        logger.info("Webull scheduler thread started")
        try:
            asyncio.run(scheduler_main())
        except Exception as e:
            logger.error("Webull scheduler crashed: %s", e, exc_info=True)

    t = threading.Thread(target=_run, daemon=True, name="webull_scheduler")
    t.start()
    logger.info("Webull scheduler thread launched")
