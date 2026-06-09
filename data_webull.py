# ─────────────────────────────────────────────
#  data_webull.py  –  Webull Open API client
#
#  Uses official Webull Open API (developer.webull.com)
#  HMAC-SHA256 signing, batch quote endpoints.
#  Never uses unofficial webull package.
#  Never hard-codes keys.
# ─────────────────────────────────────────────
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import base64
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from config import (
    WEBULL_APP_KEY, WEBULL_APP_SECRET, WEBULL_ENABLED,
    WEBULL_PAPER_TRADING, LIVE_TRADING_ENABLED,
    REQUEST_MAX_RETRIES, REQUEST_RETRY_BACKOFF,
)

logger = logging.getLogger("data_webull")

# ── Webull Open API base URLs ─────────────────────────────────────────────────
_BASE_QUOTE  = "https://openapi.webull.com"
_BASE_TRADE  = "https://openapi.webull.com"
_BASE_PAPER  = "https://openapi.webull.com"   # paper uses same base, different path

# ── HMAC signing ─────────────────────────────────────────────────────────────

def _timestamp_ms() -> str:
    return str(int(time.time() * 1000))


def _sign(app_secret: str, timestamp: str, path: str, body: str = "") -> str:
    """
    Webull Open API HMAC-SHA256 signing.
    Signature = Base64( HMAC-SHA256( app_secret, timestamp + path + body ) )
    """
    message = timestamp + path + body
    sig = hmac.new(
        app_secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.b64encode(sig).decode("utf-8")


def _headers(app_key: str, app_secret: str, path: str, body: str = "") -> Dict[str, str]:
    ts = _timestamp_ms()
    return {
        "Content-Type":   "application/json",
        "App-Key":        app_key,
        "Timestamp":      ts,
        "Sign":           _sign(app_secret, ts, path + body),
    }


# ── Paper-trade simulator ─────────────────────────────────────────────────────

class _PaperSimulator:
    """
    Minimal in-memory paper trade simulator.
    Used when LIVE_TRADING_ENABLED=False.
    """
    def __init__(self):
        self._orders: Dict[str, Dict] = {}
        self._next_id = 1

    def place(self, payload: Dict) -> Dict:
        oid = f"PAPER-{self._next_id:06d}"
        self._next_id += 1
        order = {**payload, "order_id": oid, "status": "FILLED",
                 "filled_at": datetime.now(timezone.utc).isoformat()}
        self._orders[oid] = order
        logger.info("PAPER ORDER: %s %s x%s @ %s → %s",
                    payload.get("side"), payload.get("ticker"),
                    payload.get("qty"), payload.get("limit_price"), oid)
        return order

    def cancel(self, order_id: str) -> Dict:
        if order_id in self._orders:
            self._orders[order_id]["status"] = "CANCELLED"
        return {"order_id": order_id, "status": "CANCELLED"}

    def modify(self, order_id: str, updates: Dict) -> Dict:
        if order_id in self._orders:
            self._orders[order_id].update(updates)
        return self._orders.get(order_id, {})

    def get_orders(self) -> List[Dict]:
        return list(self._orders.values())


# ── Main Webull client ────────────────────────────────────────────────────────

class WebullClient:
    """
    Official Webull Open API client.
    Handles HMAC signing, rate-limit backoff, batch quotes,
    intraday candles, options chain, and order management.
    """

    def __init__(
        self,
        app_key:    str = WEBULL_APP_KEY,
        app_secret: str = WEBULL_APP_SECRET,
        paper:      bool = WEBULL_PAPER_TRADING,
    ):
        if not app_key or not app_secret:
            logger.warning(
                "WebullClient: WEBULL_APP_KEY / WEBULL_APP_SECRET not set. "
                "Real-time data disabled."
            )
        self._key    = app_key
        self._secret = app_secret
        self._paper  = paper
        self._session = requests.Session()
        self._simulator = _PaperSimulator()
        self._enabled = WEBULL_ENABLED and bool(app_key) and bool(app_secret)

        # auth state
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0

        if self._enabled:
            self.authenticate()

    # ── Authentication ───────────────────────────────────────────────────────

    def authenticate(self) -> bool:
        """
        Authenticate with Webull Open API using client_credentials.
        Stores access_token for subsequent requests.
        """
        if not self._enabled:
            return False

        path = "/auth/token"
        payload = {
            "appKey": self._key,
            "appSecret": self._secret,
            "grantType": "client_credentials"
        }
        body = json.dumps(payload, separators=(",", ":"))
        url = _BASE_QUOTE + path
        headers = {"Content-Type": "application/json"}

        try:
            resp = self._session.post(url, headers=headers, data=body, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            self._access_token = data.get("accessToken")
            expires_in = int(data.get("expiresIn", 3600))
            self._token_expiry = time.time() + expires_in

            if self._access_token:
                logger.info("WebullClient: authenticated successfully")
                return True
            else:
                logger.error("WebullClient: authentication failed: %s", data)
                return False
        except Exception as e:
            logger.error("WebullClient: authentication error: %s", e)
            return False

    def _ensure_token(self) -> None:
        if not self._enabled:
            return
        if not self._access_token or time.time() >= self._token_expiry:
            self.authenticate()

    # ── Internal HTTP helpers ─────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        if not self._enabled:
            return {}
        self._ensure_token()
        url = _BASE_QUOTE + path
        hdrs = _headers(self._key, self._secret, path)
        if self._access_token:
            hdrs["Access-Token"] = self._access_token
        for attempt in range(REQUEST_MAX_RETRIES):
            try:
                resp = self._session.get(url, headers=hdrs, params=params, timeout=8)
                if resp.status_code == 429:
                    wait = REQUEST_RETRY_BACKOFF ** (attempt + 1)
                    logger.warning("Webull rate limit — waiting %.1fs", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and data.get("code") not in (None, 0, "0"):
                    logger.warning("Webull API error: %s", data.get("msg", data))
                return data
            except Exception as e:
                logger.debug("Webull GET %s attempt %d: %s", path, attempt + 1, e)
                if attempt < REQUEST_MAX_RETRIES - 1:
                    time.sleep(REQUEST_RETRY_BACKOFF ** attempt)
        return {}

    def _post(self, path: str, payload: Dict) -> Any:
        if not self._enabled:
            return {}
        self._ensure_token()
        url   = _BASE_TRADE + path
        body  = json.dumps(payload, separators=(",", ":"))
        hdrs  = _headers(self._key, self._secret, path, body)
        if self._access_token:
            hdrs["Access-Token"] = self._access_token
        for attempt in range(REQUEST_MAX_RETRIES):
            try:
                resp = self._session.post(url, headers=hdrs, data=body, timeout=10)
                if resp.status_code == 429:
                    wait = REQUEST_RETRY_BACKOFF ** (attempt + 1)
                    logger.warning("Webull rate limit — waiting %.1fs", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                logger.debug("Webull POST %s attempt %d: %s", path, attempt + 1, e)
                if attempt < REQUEST_MAX_RETRIES - 1:
                    time.sleep(REQUEST_RETRY_BACKOFF ** attempt)
        return {}

    # ── Quote endpoints ───────────────────────────────────────────────────────

    def get_quote(self, ticker: str) -> Dict:
        """
        Real-time quote for a single ticker.
        Returns: {ticker, last, bid, ask, volume, change_pct,
                  pre_market_price, pre_market_change_pct,
                  post_market_price, timestamp}
        """
        path = f"/quotes/v2/ticker/realtime"
        raw  = self._get(path, params={"tickerSymbol": ticker, "regionId": "6"})
        return self._parse_quote(ticker, raw)

    def get_quotes_batch(self, tickers: List[str]) -> Dict[str, Dict]:
        """
        Batch real-time quotes — much more efficient than single calls.
        Returns {ticker: quote_dict}
        """
        if not tickers:
            return {}
        results: Dict[str, Dict] = {}

        # Webull batch endpoint accepts up to 50 tickers at a time
        chunk_size = 50
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            symbols = ",".join(chunk)
            path = "/quotes/v2/ticker/realtime/list"
            raw  = self._get(path, params={"tickerSymbolList": symbols, "regionId": "6"})
            items = raw.get("data", raw) if isinstance(raw, dict) else raw
            if isinstance(items, list):
                for item in items:
                    sym = item.get("tickerSymbol") or item.get("symbol", "")
                    if sym:
                        results[sym] = self._parse_quote(sym, item)
        return results

    def _parse_quote(self, ticker: str, raw: Any) -> Dict:
        """Normalize a Webull quote response to a standard dict."""
        if not raw or isinstance(raw, dict) and not raw:
            return {"ticker": ticker, "available": False}

        # Webull returns data nested under "data" or directly
        d = raw.get("data", raw) if isinstance(raw, dict) else {}
        if isinstance(d, list) and d:
            d = d[0]

        def _f(key: str, default=0.0) -> float:
            v = d.get(key, default)
            try: return float(v) if v not in (None, "", "null") else default
            except: return default

        return {
            "ticker":               ticker,
            "available":            True,
            "last":                 _f("close") or _f("last") or _f("latestPrice"),
            "bid":                  _f("bidPrice"),
            "ask":                  _f("askPrice"),
            "volume":               _f("volume"),
            "change_pct":           _f("changeRatio"),
            "change_abs":           _f("change"),
            "high":                 _f("high"),
            "low":                  _f("low"),
            "open":                 _f("open"),
            "pre_market_price":     _f("preMarketPrice") or _f("extendedHoursPrice"),
            "pre_market_chg_pct":   _f("preMarketChangeRatio"),
            "post_market_price":    _f("afterHoursPrice"),
            "post_market_chg_pct":  _f("afterHoursChangeRatio"),
            "timestamp":            d.get("tradeTime") or _timestamp_ms(),
        }

    # ── Intraday candles ──────────────────────────────────────────────────────

    def get_intraday_candles(
        self,
        ticker: str,
        interval: str = "m1",
        lookback_minutes: int = 60,
    ) -> List[Dict]:
        """
        Recent intraday OHLCV candles.
        interval: m1, m5, m15, m30, h1
        Returns list of {time, open, high, low, close, volume}
        """
        path = "/quotes/v2/ticker/charts"
        raw  = self._get(path, params={
            "tickerSymbol": ticker,
            "type":         interval,
            "count":        lookback_minutes,
            "regionId":     "6",
        })
        items = raw.get("data", []) if isinstance(raw, dict) else []
        candles = []
        for item in items:
            try:
                candles.append({
                    "time":   item.get("id") or item.get("timestamp"),
                    "open":   float(item.get("open",  0)),
                    "high":   float(item.get("high",  0)),
                    "low":    float(item.get("low",   0)),
                    "close":  float(item.get("close", 0)),
                    "volume": float(item.get("volume", 0)),
                })
            except (TypeError, ValueError):
                continue
        return candles

    # ── Options chain ─────────────────────────────────────────────────────────

    def get_options_chain(self, ticker: str) -> Dict:
        """
        Full options chain with greeks, IV, OI.
        Returns {ticker, price, expirations: [{expiry, calls, puts}]}
        """
        path = "/quotes/v2/option/list"
        raw  = self._get(path, params={"tickerSymbol": ticker, "regionId": "6"})
        data = raw.get("data", {}) if isinstance(raw, dict) else {}

        expirations = []
        for exp_group in data.get("expireDateList", []):
            expiry = exp_group.get("expireDate", "")
            calls, puts = [], []
            for opt in exp_group.get("callList", []):
                calls.append(self._parse_option(opt))
            for opt in exp_group.get("putList", []):
                puts.append(self._parse_option(opt))
            expirations.append({
                "expiry": expiry,
                "calls":  calls,
                "puts":   puts,
            })

        return {
            "ticker":      ticker,
            "price":       float(data.get("lastClosePrice", 0)),
            "expirations": expirations,
        }

    def _parse_option(self, opt: Dict) -> Dict:
        def _f(k): 
            try: return float(opt.get(k, 0) or 0)
            except: return 0.0
        return {
            "strike":          _f("strikePrice"),
            "last":            _f("close"),
            "bid":             _f("bidPrice"),
            "ask":             _f("askPrice"),
            "volume":          int(_f("volume")),
            "open_interest":   int(_f("openInterest")),
            "iv":              _f("impliedVolatility"),
            "delta":           _f("delta"),
            "gamma":           _f("gamma"),
            "theta":           _f("theta"),
            "vega":            _f("vega"),
        }

    # ── Order management ──────────────────────────────────────────────────────

    def place_order(
        self,
        ticker:      str,
        side:        str,          # "BUY" | "SELL"
        qty:         int,
        order_type:  str,          # "LMT" | "MKT" | "STP" | "STP_LMT"
        limit_price: Optional[float] = None,
        stop_price:  Optional[float] = None,
        bracket:     Optional[Dict]  = None,
    ) -> Dict:
        """
        Place an order. Routes to paper simulator unless LIVE_TRADING_ENABLED.
        bracket: {"stop_loss": float, "take_profit": float}
        """
        if not LIVE_TRADING_ENABLED:
            logger.info("PAPER mode — simulating order for %s", ticker)
            return self._simulator.place({
                "ticker": ticker, "side": side, "qty": qty,
                "order_type": order_type, "limit_price": limit_price,
                "stop_price": stop_price, "bracket": bracket,
            })

        payload = {
            "tickerSymbol": ticker,
            "side":         side,
            "orderType":    order_type,
            "qty":          qty,
            "regionId":     "6",
            "timeInForce":  "DAY",
        }
        if limit_price: payload["lmtPrice"] = str(limit_price)
        if stop_price:  payload["auxPrice"]  = str(stop_price)

        if bracket:
            payload["bracketOrder"] = {
                "stopLoss":   str(bracket["stop_loss"]),
                "takeProfit": str(bracket["take_profit"]),
            }

        path = "/trade/v2/order/place"
        return self._post(path, payload)

    def cancel_order(self, order_id: str) -> Dict:
        if not LIVE_TRADING_ENABLED:
            return self._simulator.cancel(order_id)
        path = "/trade/v2/order/cancel"
        return self._post(path, {"orderId": order_id})

    def modify_order(self, order_id: str, updates: Dict) -> Dict:
        if not LIVE_TRADING_ENABLED:
            return self._simulator.modify(order_id, updates)
        path = "/trade/v2/order/modify"
        return self._post(path, {"orderId": order_id, **updates})

    def get_positions(self) -> List[Dict]:
        """Sync open positions from Webull account."""
        if not LIVE_TRADING_ENABLED:
            return []
        path = "/trade/v2/position/list"
        raw  = self._get(path)
        positions = raw.get("data", []) if isinstance(raw, dict) else []
        return [
            {
                "ticker":    p.get("tickerSymbol", ""),
                "qty":       float(p.get("position", 0)),
                "avg_cost":  float(p.get("costPrice", 0)),
                "mkt_value": float(p.get("marketValue", 0)),
                "pnl":       float(p.get("unrealizedProfitLoss", 0)),
            }
            for p in positions if isinstance(p, dict)
        ]

    def get_account(self) -> Dict:
        """Account summary: buying power, net liquidation."""
        if not LIVE_TRADING_ENABLED:
            return {"buying_power": 0, "net_liq": 0, "paper": True}
        raw = self._get("/trade/v2/account/summary")
        d   = raw.get("data", {}) if isinstance(raw, dict) else {}
        return {
            "buying_power": float(d.get("buyingPower", 0)),
            "net_liq":      float(d.get("netLiquidation", 0)),
            "paper":        self._paper,
        }

    def is_available(self) -> bool:
        return self._enabled


# ── Module-level singleton ────────────────────────────────────────────────────
_client: Optional[WebullClient] = None

def get_client() -> WebullClient:
    global _client
    if _client is None:
        _client = WebullClient()
    return _client
