# ─────────────────────────────────────────────
#  history.py  –  Persistent HistoryStore
#
#  Saves to disk (JSON) so score history and
#  alert logs survive Railway restarts.
# ─────────────────────────────────────────────
from __future__ import annotations
import datetime
import json
import logging
import os
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from config import HISTORY_CONFIG

logger = logging.getLogger("history")

_PERSISTENCE_FILE = HISTORY_CONFIG.get("persistence_file", "history_store.json")
_ALERT_LOG_FILE   = HISTORY_CONFIG.get("alert_log_file",   "alert_history.jsonl")
_MAX_PER_KEY      = HISTORY_CONFIG.get("max_history_per_key", 90)


class HistoryStore:
    """
    Persistent in-memory store.

    Schema
    ------
    _stock[ticker][key]  = [(iso_date_str, value), ...]
    _market[key]         = [(iso_date_str, value), ...]
    _score_history[ticker] = [(iso_date_str, upside_score), ...]  sparkline data
    """

    def __init__(self):
        self._lock   = threading.Lock()
        self._stock  : Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
        self._market : Dict[str, List]             = defaultdict(list)
        self._score_history: Dict[str, List]       = defaultdict(list)
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        if not os.path.exists(_PERSISTENCE_FILE):
            return
        try:
            with open(_PERSISTENCE_FILE, "r") as f:
                data = json.load(f)
            for ticker, keys in data.get("stock", {}).items():
                for key, records in keys.items():
                    self._stock[ticker][key] = [tuple(r) for r in records]
            for key, records in data.get("market", {}).items():
                self._market[key] = [tuple(r) for r in records]
            for ticker, records in data.get("score_history", {}).items():
                self._score_history[ticker] = [tuple(r) for r in records]
            logger.info("HistoryStore loaded from %s", _PERSISTENCE_FILE)
        except Exception as e:
            logger.warning("Could not load history: %s", e)

    def save(self):
        """Write current state to disk. Called after each scan."""
        try:
            with self._lock:
                data = {
                    "stock":         {t: {k: list(v) for k, v in keys.items()}
                                      for t, keys in self._stock.items()},
                    "market":        {k: list(v) for k, v in self._market.items()},
                    "score_history": {t: list(v) for t, v in self._score_history.items()},
                    "saved_at":      datetime.datetime.utcnow().isoformat(),
                }
            with open(_PERSISTENCE_FILE, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning("Could not save history: %s", e)

    # ── Alert log ─────────────────────────────────────────────────────────────

    def log_alert(self, alert: Dict[str, Any]):
        """Append a fired alert to the persistent JSONL log."""
        try:
            record = {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "ticker":    alert.get("ticker"),
                "upside":    round(float(alert.get("upside", 0)), 3),
                "risk":      round(float(alert.get("risk",   0)), 3),
                "regime":    alert.get("regime"),
                "setups":    alert.get("setups", []),
                "sector":    alert.get("sector"),
                "price":     alert.get("last_price"),
            }
            with open(_ALERT_LOG_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning("Could not log alert: %s", e)

    def get_alert_history(self, limit: int = 200) -> List[Dict]:
        """Return the most recent *limit* alerts from the log file."""
        if not os.path.exists(_ALERT_LOG_FILE):
            return []
        try:
            records = []
            with open(_ALERT_LOG_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records[-limit:][::-1]   # most recent first
        except Exception as e:
            logger.warning("Could not read alert history: %s", e)
            return []

    # ── Stock-level ───────────────────────────────────────────────────────────

    def update_stock(self, ticker: str, date: Optional[datetime.date], data: Dict[str, Any]):
        if date is None:
            date = datetime.date.today()
        date_str = date.isoformat()
        with self._lock:
            for key, value in data.items():
                try:
                    fval = float(value)
                except (TypeError, ValueError):
                    continue
                records = self._stock[ticker][key]
                records.append((date_str, fval))
                # Keep only most recent N
                if len(records) > _MAX_PER_KEY:
                    self._stock[ticker][key] = records[-_MAX_PER_KEY:]

            # Maintain separate sparkline history for upside
            if "upside" in data:
                try:
                    uval = float(data["upside"])
                    sh = self._score_history[ticker]
                    sh.append((date_str, uval))
                    if len(sh) > _MAX_PER_KEY:
                        self._score_history[ticker] = sh[-_MAX_PER_KEY:]
                except (TypeError, ValueError):
                    pass

    def get_stock_history(self, ticker: str, key: str, lookback_days: int = 60) -> List[float]:
        cutoff = (datetime.date.today() - datetime.timedelta(days=lookback_days)).isoformat()
        with self._lock:
            records = self._stock[ticker].get(key, [])
            return [v for d, v in records if d >= cutoff]

    def get_score_sparkline(self, ticker: str, lookback_days: int = 30) -> List[Dict]:
        """Return [{date, score}] for sparkline rendering."""
        cutoff = (datetime.date.today() - datetime.timedelta(days=lookback_days)).isoformat()
        with self._lock:
            records = self._score_history.get(ticker, [])
            return [{"date": d, "score": round(v, 3)}
                    for d, v in records if d >= cutoff]

    def get_all_stock_values(self, key: str, lookback_days: int = 60) -> List[float]:
        cutoff = (datetime.date.today() - datetime.timedelta(days=lookback_days)).isoformat()
        values = []
        with self._lock:
            for ticker_data in self._stock.values():
                values.extend(v for d, v in ticker_data.get(key, []) if d >= cutoff)
        return values

    # ── Market-level ──────────────────────────────────────────────────────────

    def update_market(self, date: Optional[datetime.date], data: Dict[str, Any]):
        if date is None:
            date = datetime.date.today()
        date_str = date.isoformat()
        with self._lock:
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    self._market[key].append((date_str, float(value)))
                else:
                    self._market[key].append((date_str, value))

    def get_market_history(self, key: str, lookback_days: int = 90) -> List:
        cutoff = (datetime.date.today() - datetime.timedelta(days=lookback_days)).isoformat()
        with self._lock:
            records = self._market.get(key, [])
            return [v for d, v in records if d >= cutoff]

    # ── Statistical helpers ───────────────────────────────────────────────────

    def percentile_rank(self, value: float, series: List[float]) -> float:
        if not series:
            return 0.5
        arr = np.array(series, dtype=float)
        return float(np.mean(arr <= value))

    def zscore(self, value: float, series: List[float]) -> float:
        if len(series) < 2:
            return 0.0
        arr = np.array(series, dtype=float)
        mu, sigma = arr.mean(), arr.std()
        if sigma < 1e-9:
            return 0.0
        return float(np.clip((value - mu) / sigma, -3.0, 3.0))

    def zscore_to_score(self, zscore_val: float) -> float:
        return float(np.clip((zscore_val + 3.0) / 6.0, 0.0, 1.0))

    def rolling_correlation(self, factor_key: str, return_key: str,
                             ticker: str, window: int = 30) -> float:
        f_vals = self.get_stock_history(ticker, factor_key, lookback_days=window)
        r_vals = self.get_stock_history(ticker, return_key,  lookback_days=window)
        n = min(len(f_vals), len(r_vals))
        if n < 5:
            return 0.0
        f = np.array(f_vals[-n:])
        r = np.array(r_vals[-n:])
        if f.std() < 1e-9 or r.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(f, r)[0, 1])

    def cross_sectional_rolling_corr(self, factor_key: str, return_key: str,
                                      window: int = 30) -> float:
        corrs = []
        with self._lock:
            tickers = list(self._stock.keys())
        for ticker in tickers:
            c = self.rolling_correlation(factor_key, return_key, ticker, window)
            corrs.append(c)
        return float(np.mean(corrs)) if corrs else 0.0


# ── Singleton ─────────────────────────────────────────────────────────────────
# Shared instance used by app.py so history persists across scans
_global_store: Optional[HistoryStore] = None


def get_global_store() -> HistoryStore:
    global _global_store
    if _global_store is None:
        _global_store = HistoryStore()
    return _global_store
