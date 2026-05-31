# ─────────────────────────────────────────────
#  storage/history.py  –  In-memory HistoryStore
# ─────────────────────────────────────────────
from __future__ import annotations
import datetime
from collections import defaultdict
from typing import Any, Dict, List, Optional
import numpy as np


class HistoryStore:
    """
    Thread-safe(ish) in-memory store for:
      - Per-ticker factor scores & UpsideScores over time
      - Market-wide data (regimes, breadth, etc.)
      - Methods for percentile/z-score normalization
      - Rolling correlation helper for adaptive weights

    Schema
    ------
    _stock[ticker][key] = [(date, value), ...]   sorted ascending by date
    _market[key]        = [(date, value), ...]   sorted ascending by date
    """

    def __init__(self):
        # defaultdict of defaultdict of list
        self._stock: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
        self._market: Dict[str, List] = defaultdict(list)

    # ── Stock-level ──────────────────────────────────────────────────────────

    def update_stock(self, ticker: str, date: Optional[datetime.date], data: Dict[str, Any]):
        """Store a snapshot of scores/factors for a ticker on a given date."""
        if date is None:
            date = datetime.date.today()
        for key, value in data.items():
            self._stock[ticker][key].append((date, float(value)))

    def get_stock_history(
        self,
        ticker: str,
        key: str,
        lookback_days: int = 60,
    ) -> List[float]:
        """Return the most recent `lookback_days` values for (ticker, key)."""
        records = self._stock[ticker].get(key, [])
        if not records:
            return []
        cutoff = datetime.date.today() - datetime.timedelta(days=lookback_days)
        values = [v for d, v in records if d >= cutoff]
        return values

    def get_all_stock_values(self, key: str, lookback_days: int = 60) -> List[float]:
        """Aggregate a key across ALL tickers (useful for cross-sectional percentile)."""
        values = []
        cutoff = datetime.date.today() - datetime.timedelta(days=lookback_days)
        for ticker_data in self._stock.values():
            values.extend(v for d, v in ticker_data.get(key, []) if d >= cutoff)
        return values

    # ── Market-level ─────────────────────────────────────────────────────────

    def update_market(self, date: Optional[datetime.date], data: Dict[str, Any]):
        if date is None:
            date = datetime.date.today()
        for key, value in data.items():
            if isinstance(value, (int, float)):
                self._market[key].append((date, float(value)))
            else:
                self._market[key].append((date, value))

    def get_market_history(self, key: str, lookback_days: int = 90) -> List:
        records = self._market.get(key, [])
        if not records:
            return []
        cutoff = datetime.date.today() - datetime.timedelta(days=lookback_days)
        return [v for d, v in records if d >= cutoff]

    # ── Statistical helpers ───────────────────────────────────────────────────

    def percentile_rank(self, value: float, series: List[float]) -> float:
        """Return what fraction of `series` is <= value (0–1)."""
        if not series:
            return 0.5
        arr = np.array(series, dtype=float)
        return float(np.mean(arr <= value))

    def zscore(self, value: float, series: List[float]) -> float:
        """Return z-score of value relative to series; clipped to [-3, 3]."""
        if len(series) < 2:
            return 0.0
        arr = np.array(series, dtype=float)
        mu, sigma = arr.mean(), arr.std()
        if sigma < 1e-9:
            return 0.0
        return float(np.clip((value - mu) / sigma, -3.0, 3.0))

    def zscore_to_score(self, zscore_val: float) -> float:
        """Map a z-score in [-3, 3] to a 0–1 score via sigmoid-like mapping."""
        return float(np.clip((zscore_val + 3.0) / 6.0, 0.0, 1.0))

    def rolling_correlation(
        self,
        factor_key: str,
        return_key: str,
        ticker: str,
        window: int = 30,
    ) -> float:
        """
        Compute Pearson correlation between factor_key and return_key
        for a given ticker over the last `window` days.
        Returns 0.0 if insufficient data.
        """
        f_vals = self.get_stock_history(ticker, factor_key, lookback_days=window)
        r_vals = self.get_stock_history(ticker, return_key, lookback_days=window)
        n = min(len(f_vals), len(r_vals))
        if n < 5:
            return 0.0
        f = np.array(f_vals[-n:])
        r = np.array(r_vals[-n:])
        if f.std() < 1e-9 or r.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(f, r)[0, 1])

    def cross_sectional_rolling_corr(
        self,
        factor_key: str,
        return_key: str,
        window: int = 30,
    ) -> float:
        """
        Average rolling correlation across all tickers in the store.
        Used for adaptive weight estimation.
        """
        corrs = []
        for ticker in self._stock:
            c = self.rolling_correlation(factor_key, return_key, ticker, window)
            corrs.append(c)
        if not corrs:
            return 0.0
        return float(np.mean(corrs))
