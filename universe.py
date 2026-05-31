# ─────────────────────────────────────────────
#  data/universe.py  –  Ticker universe
#
#  Real implementation: pull from your broker /
#  data provider and apply basic filters.
# ─────────────────────────────────────────────
from __future__ import annotations
from config import UNIVERSE_CONFIG


# Demo set – a cross-section of real tickers across sectors & caps
_DEMO_TICKERS = [
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


def get_universe() -> list[str]:
    """
    Return the list of tickers to scan.

    Priority:
      1. If UNIVERSE_CONFIG["ticker_list"] is set, use it.
      2. Otherwise return the built-in demo set.

    Real implementation should:
      - Pull from an exchange listing or your broker's screener
      - Filter by min_price, min_avg_dollar_volume, exchange, etc.
    """
    explicit = UNIVERSE_CONFIG.get("ticker_list")
    if explicit:
        return list(explicit)

    return list(_DEMO_TICKERS)
