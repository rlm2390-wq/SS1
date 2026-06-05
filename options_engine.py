# ─────────────────────────────────────────────
#  options_engine.py  –  Options play generator
#
#  Analyzes the options chain for a ticker and
#  recommends specific directional or premium-
#  collection plays with strikes, expirations,
#  and rough P&L estimates.
# ─────────────────────────────────────────────
from __future__ import annotations
import datetime
import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

from config import OPTIONS_CONFIG

logger = logging.getLogger("options_engine")

_MIN_DTE = OPTIONS_CONFIG["min_days_to_expiry"]
_MAX_DTE = OPTIONS_CONFIG["max_days_to_expiry"]
_LOW_IV  = OPTIONS_CONFIG["low_iv_threshold"]
_HIGH_IV = OPTIONS_CONFIG["high_iv_threshold"]
_MIN_OI  = OPTIONS_CONFIG["min_open_interest"]


# ── Black-Scholes helpers ─────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _bs_call_price(S, K, T, r, sigma) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def _bs_put_price(S, K, T, r, sigma) -> float:
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


# ── Chain helpers ─────────────────────────────────────────────────────────────

def _get_chain(ticker: str) -> Optional[Dict]:
    """Fetch the options chain and return a structured dict."""
    try:
        t    = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return None

        today = datetime.date.today()
        best_exp = None
        best_dte = None
        for exp in exps:
            dte = (datetime.date.fromisoformat(exp) - today).days
            if _MIN_DTE <= dte <= _MAX_DTE:
                if best_dte is None or abs(dte - 30) < abs(best_dte - 30):
                    best_exp = exp
                    best_dte = dte

        if not best_exp:
            best_exp = exps[0]
            best_dte = (datetime.date.fromisoformat(best_exp) - today).days

        chain  = t.option_chain(best_exp)
        calls  = chain.calls
        puts   = chain.puts
        price  = float(t.history(period="1d")["Close"].iloc[-1])

        # IV rank proxy (use ATM IV)
        atm_calls = calls[abs(calls["strike"] - price) == abs(calls["strike"] - price).min()]
        iv_atm    = float(atm_calls["impliedVolatility"].mean() * 100) if len(atm_calls) else 30.0
        iv_rank   = min(iv_atm, 100.0)

        return {
            "ticker":   ticker,
            "price":    price,
            "exp":      best_exp,
            "dte":      best_dte,
            "iv_rank":  iv_rank,
            "calls":    calls,
            "puts":     puts,
        }
    except Exception as e:
        logger.warning("get_chain failed for %s: %s", ticker, e)
        return None


def _find_strike(df, price: float, offset_pct: float = 0.0, side: str = "call"):
    """Find the nearest strike to price * (1 + offset_pct)."""
    target = price * (1 + offset_pct)
    df     = df.copy()
    df["dist"] = abs(df["strike"] - target)
    row = df.nsmallest(1, "dist").iloc[0]
    return row


# ── Play builders ─────────────────────────────────────────────────────────────

def _long_call(chain: Dict, result: Dict) -> Optional[Dict]:
    price    = chain["price"]
    dte      = chain["dte"]
    iv_rank  = chain["iv_rank"]
    calls    = chain["calls"]
    T        = dte / 365.0

    # ATM or slight OTM (+2%)
    row = _find_strike(calls, price, offset_pct=0.02)
    strike   = float(row["strike"])
    last     = float(row.get("lastPrice", 0))
    oi       = int(row.get("openInterest", 0))
    if oi < _MIN_OI or last <= 0:
        return None

    entry       = round(last * 100, 2)        # cost per contract
    max_loss    = entry
    max_profit  = None                         # theoretically unlimited
    breakeven   = round(strike + last, 2)

    return {
        "strategy":    "Long Call",
        "type":        "debit",
        "direction":   "bullish",
        "ticker":      chain["ticker"],
        "price":       price,
        "expiry":      chain["exp"],
        "dte":         dte,
        "iv_rank":     round(iv_rank, 1),
        "strike":      strike,
        "entry_cost":  entry,
        "max_loss":    max_loss,
        "max_profit":  "Unlimited",
        "breakeven":   breakeven,
        "explanation": (
            f"Buy the {strike} call expiring {chain['exp']}. "
            f"You pay ${entry:.0f}/contract. Stock needs to close above "
            f"${breakeven:.2f} by expiry to profit. "
            f"IV rank is {iv_rank:.0f} — options are {'cheap' if iv_rank < 35 else 'moderate'}, "
            f"making debit plays {'attractive' if iv_rank < 35 else 'reasonable'}."
        ),
        "risk_defined": True,
    }


def _long_put(chain: Dict, result: Dict) -> Optional[Dict]:
    price   = chain["price"]
    dte     = chain["dte"]
    iv_rank = chain["iv_rank"]
    puts    = chain["puts"]

    row = _find_strike(puts, price, offset_pct=-0.02)
    strike  = float(row["strike"])
    last    = float(row.get("lastPrice", 0))
    oi      = int(row.get("openInterest", 0))
    if oi < _MIN_OI or last <= 0:
        return None

    entry     = round(last * 100, 2)
    breakeven = round(strike - last, 2)

    return {
        "strategy":    "Long Put",
        "type":        "debit",
        "direction":   "bearish",
        "ticker":      chain["ticker"],
        "price":       price,
        "expiry":      chain["exp"],
        "dte":         dte,
        "iv_rank":     round(iv_rank, 1),
        "strike":      strike,
        "entry_cost":  entry,
        "max_loss":    entry,
        "max_profit":  round((strike - last) * 100, 2),
        "breakeven":   breakeven,
        "explanation": (
            f"Buy the {strike} put expiring {chain['exp']}. "
            f"Pay ${entry:.0f}/contract. Profits if stock falls below ${breakeven:.2f}. "
            f"Bearish bet with defined risk."
        ),
        "risk_defined": True,
    }


def _call_debit_spread(chain: Dict, result: Dict) -> Optional[Dict]:
    price   = chain["price"]
    dte     = chain["dte"]
    iv_rank = chain["iv_rank"]
    calls   = chain["calls"]
    width   = min(OPTIONS_CONFIG["max_spread_width"], round(price * 0.05))

    buy_row  = _find_strike(calls, price, offset_pct=0.01)
    sell_row = _find_strike(calls, price, offset_pct=0.06)
    buy_str  = float(buy_row["strike"])
    sell_str = float(sell_row["strike"])
    buy_last = float(buy_row.get("lastPrice", 0))
    sell_last= float(sell_row.get("lastPrice", 0))
    if buy_last <= sell_last or buy_last <= 0:
        return None

    debit      = round((buy_last - sell_last) * 100, 2)
    max_profit = round((sell_str - buy_str - (buy_last - sell_last)) * 100, 2)
    breakeven  = round(buy_str + (buy_last - sell_last), 2)

    return {
        "strategy":    "Call Debit Spread",
        "type":        "debit",
        "direction":   "bullish",
        "ticker":      chain["ticker"],
        "price":       price,
        "expiry":      chain["exp"],
        "dte":         dte,
        "iv_rank":     round(iv_rank, 1),
        "strike":      f"{buy_str}/{sell_str}",
        "entry_cost":  debit,
        "max_loss":    debit,
        "max_profit":  max_profit,
        "breakeven":   breakeven,
        "explanation": (
            f"Buy {buy_str} call, sell {sell_str} call, same expiry {chain['exp']}. "
            f"Pay ${debit:.0f}/contract. Max profit ${max_profit:.0f} if stock closes "
            f"above ${sell_str}. Cheaper than a naked call because you sell the upper strike. "
            f"IV rank {iv_rank:.0f} is elevated — spread reduces IV risk."
        ),
        "risk_defined": True,
    }


def _put_debit_spread(chain: Dict, result: Dict) -> Optional[Dict]:
    price    = chain["price"]
    dte      = chain["dte"]
    iv_rank  = chain["iv_rank"]
    puts     = chain["puts"]

    buy_row  = _find_strike(puts, price, offset_pct=-0.01)
    sell_row = _find_strike(puts, price, offset_pct=-0.06)
    buy_str  = float(buy_row["strike"])
    sell_str = float(sell_row["strike"])
    buy_last = float(buy_row.get("lastPrice", 0))
    sell_last= float(sell_row.get("lastPrice", 0))
    if buy_last <= sell_last or buy_last <= 0:
        return None

    debit      = round((buy_last - sell_last) * 100, 2)
    max_profit = round((buy_str - sell_str - (buy_last - sell_last)) * 100, 2)
    breakeven  = round(buy_str - (buy_last - sell_last), 2)

    return {
        "strategy":    "Put Debit Spread",
        "type":        "debit",
        "direction":   "bearish",
        "ticker":      chain["ticker"],
        "price":       price,
        "expiry":      chain["exp"],
        "dte":         dte,
        "iv_rank":     round(iv_rank, 1),
        "strike":      f"{sell_str}/{buy_str}",
        "entry_cost":  debit,
        "max_loss":    debit,
        "max_profit":  max_profit,
        "breakeven":   breakeven,
        "explanation": (
            f"Buy {buy_str} put, sell {sell_str} put expiring {chain['exp']}. "
            f"Pay ${debit:.0f}/contract. Profits if stock falls below ${breakeven:.2f}. "
            f"Defined-risk bearish play."
        ),
        "risk_defined": True,
    }


def _cash_secured_put(chain: Dict, result: Dict) -> Optional[Dict]:
    price   = chain["price"]
    dte     = chain["dte"]
    iv_rank = chain["iv_rank"]
    puts    = chain["puts"]

    # Sell OTM put ~5% below price
    row    = _find_strike(puts, price, offset_pct=-0.05)
    strike = float(row["strike"])
    last   = float(row.get("lastPrice", 0))
    oi     = int(row.get("openInterest", 0))
    if oi < _MIN_OI or last <= 0:
        return None

    premium    = round(last * 100, 2)
    max_profit = premium
    max_loss   = round((strike - last) * 100, 2)
    breakeven  = round(strike - last, 2)

    return {
        "strategy":    "Cash-Secured Put",
        "type":        "credit",
        "direction":   "neutral/bullish",
        "ticker":      chain["ticker"],
        "price":       price,
        "expiry":      chain["exp"],
        "dte":         dte,
        "iv_rank":     round(iv_rank, 1),
        "strike":      strike,
        "premium":     premium,
        "max_loss":    max_loss,
        "max_profit":  max_profit,
        "breakeven":   breakeven,
        "explanation": (
            f"Sell the {strike} put expiring {chain['exp']}. "
            f"Collect ${premium:.0f}/contract upfront. "
            f"Keep all premium if stock stays above ${strike}. "
            f"If assigned, you buy shares at ${breakeven:.2f} effective cost. "
            f"High IV rank ({iv_rank:.0f}) means you're collecting rich premium."
        ),
        "risk_defined": True,
    }


def _bull_put_spread(chain: Dict, result: Dict) -> Optional[Dict]:
    price   = chain["price"]
    dte     = chain["dte"]
    iv_rank = chain["iv_rank"]
    puts    = chain["puts"]

    sell_row = _find_strike(puts, price, offset_pct=-0.04)
    buy_row  = _find_strike(puts, price, offset_pct=-0.09)
    sell_str = float(sell_row["strike"])
    buy_str  = float(buy_row["strike"])
    sell_last= float(sell_row.get("lastPrice", 0))
    buy_last = float(buy_row.get("lastPrice",  0))
    if sell_last <= buy_last or sell_last <= 0:
        return None

    credit     = round((sell_last - buy_last) * 100, 2)
    max_loss   = round((sell_str - buy_str - (sell_last - buy_last)) * 100, 2)
    breakeven  = round(sell_str - (sell_last - buy_last), 2)

    return {
        "strategy":    "Bull Put Credit Spread",
        "type":        "credit",
        "direction":   "neutral/bullish",
        "ticker":      chain["ticker"],
        "price":       price,
        "expiry":      chain["exp"],
        "dte":         dte,
        "iv_rank":     round(iv_rank, 1),
        "strike":      f"{buy_str}/{sell_str}",
        "premium":     credit,
        "max_loss":    max_loss,
        "max_profit":  credit,
        "breakeven":   breakeven,
        "explanation": (
            f"Sell {sell_str} put, buy {buy_str} put expiring {chain['exp']}. "
            f"Collect ${credit:.0f}/contract. Keep all if stock stays above ${sell_str}. "
            f"Max loss ${max_loss:.0f} if stock drops below ${buy_str}. "
            f"Defined risk — great for high-IV environments."
        ),
        "risk_defined": True,
    }


def _bear_call_spread(chain: Dict, result: Dict) -> Optional[Dict]:
    price   = chain["price"]
    dte     = chain["dte"]
    iv_rank = chain["iv_rank"]
    calls   = chain["calls"]

    sell_row = _find_strike(calls, price, offset_pct=0.04)
    buy_row  = _find_strike(calls, price, offset_pct=0.09)
    sell_str = float(sell_row["strike"])
    buy_str  = float(buy_row["strike"])
    sell_last= float(sell_row.get("lastPrice", 0))
    buy_last = float(buy_row.get("lastPrice",  0))
    if sell_last <= buy_last or sell_last <= 0:
        return None

    credit    = round((sell_last - buy_last) * 100, 2)
    max_loss  = round((buy_str - sell_str - (sell_last - buy_last)) * 100, 2)
    breakeven = round(sell_str + (sell_last - buy_last), 2)

    return {
        "strategy":    "Bear Call Credit Spread",
        "type":        "credit",
        "direction":   "neutral/bearish",
        "ticker":      chain["ticker"],
        "price":       price,
        "expiry":      chain["exp"],
        "dte":         dte,
        "iv_rank":     round(iv_rank, 1),
        "strike":      f"{sell_str}/{buy_str}",
        "premium":     credit,
        "max_loss":    max_loss,
        "max_profit":  credit,
        "breakeven":   breakeven,
        "explanation": (
            f"Sell {sell_str} call, buy {buy_str} call expiring {chain['exp']}. "
            f"Collect ${credit:.0f}/contract. Keep all if stock stays below ${sell_str}. "
            f"Profits in flat to slightly down market."
        ),
        "risk_defined": True,
    }


def _iron_condor(chain: Dict, result: Dict) -> Optional[Dict]:
    price   = chain["price"]
    dte     = chain["dte"]
    iv_rank = chain["iv_rank"]
    calls   = chain["calls"]
    puts    = chain["puts"]

    # Bull put spread legs
    sp_sell = _find_strike(puts,  price, offset_pct=-0.05)
    sp_buy  = _find_strike(puts,  price, offset_pct=-0.10)
    # Bear call spread legs
    sc_sell = _find_strike(calls, price, offset_pct=0.05)
    sc_buy  = _find_strike(calls, price, offset_pct=0.10)

    put_credit  = float(sp_sell.get("lastPrice", 0)) - float(sp_buy.get("lastPrice",  0))
    call_credit = float(sc_sell.get("lastPrice", 0)) - float(sc_buy.get("lastPrice",  0))
    if put_credit <= 0 or call_credit <= 0:
        return None

    total_credit = round((put_credit + call_credit) * 100, 2)
    put_width    = float(sp_sell["strike"]) - float(sp_buy["strike"])
    call_width   = float(sc_buy["strike"])  - float(sc_sell["strike"])
    max_loss     = round((max(put_width, call_width) - put_credit - call_credit) * 100, 2)

    return {
        "strategy":    "Iron Condor",
        "type":        "credit",
        "direction":   "neutral",
        "ticker":      chain["ticker"],
        "price":       price,
        "expiry":      chain["exp"],
        "dte":         dte,
        "iv_rank":     round(iv_rank, 1),
        "strike":      (f"{float(sp_buy['strike'])}/{float(sp_sell['strike'])} put · "
                        f"{float(sc_sell['strike'])}/{float(sc_buy['strike'])} call"),
        "premium":     total_credit,
        "max_loss":    max_loss,
        "max_profit":  total_credit,
        "breakeven":   f"{float(sp_sell['strike'])-put_credit:.2f} / {float(sc_sell['strike'])+call_credit:.2f}",
        "explanation": (
            f"Sell a put spread AND a call spread on {chain['ticker']} expiring {chain['exp']}. "
            f"Collect ${total_credit:.0f}/contract. "
            f"Profit zone: stock stays between ${float(sp_sell['strike']):.2f} and ${float(sc_sell['strike']):.2f}. "
            f"IV rank {iv_rank:.0f} is high — ideal for iron condors since you're selling overpriced IV."
        ),
        "risk_defined": True,
    }


# ── Confidence scorer ─────────────────────────────────────────────────────────

def _confidence(result: Dict, strategy_type: str, iv_rank: float) -> int:
    """Return 1–5 star confidence based on signal alignment."""
    score = 0
    upside  = result.get("upside", 0)
    setups  = result.get("setups", [])
    tech    = result.get("factor_scores", {}).get("technical", 0)
    struct  = result.get("factor_scores", {}).get("structural", 0)

    if upside > 0.70:   score += 2
    elif upside > 0.60: score += 1

    if setups:          score += 1
    if tech > 0.65:     score += 1

    if strategy_type == "debit" and iv_rank < _LOW_IV:    score += 1
    if strategy_type == "credit" and iv_rank > _HIGH_IV:  score += 1

    return min(max(score, 1), 5)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_options_plays(results: List[Dict]) -> List[Dict]:
    """
    Given a list of scan results, find the best options plays.
    Returns list of play dicts ready for the dashboard.
    """
    if not YF_AVAILABLE:
        return []

    plays = []
    # Use top 20 results for options analysis
    candidates = sorted(results, key=lambda x: x.get("upside", 0), reverse=True)[:20]

    for result in candidates:
        ticker  = result.get("ticker")
        upside  = result.get("upside", 0)
        risk    = result.get("risk",   1)
        regime  = result.get("regime", "unknown")
        setups_list = result.get("setups", [])

        if upside < 0.50 or risk > 0.70:
            continue

        chain = _get_chain(ticker)
        if not chain:
            continue

        iv_rank = chain["iv_rank"]
        play    = None

        # Strategy selection logic
        is_bullish  = upside >= 0.60 and result.get("factor_scores", {}).get("technical", 0) >= 0.55
        is_bearish  = result.get("risk", 0) > 0.55 and upside < 0.55
        is_choppy   = regime in ("choppy",)
        low_iv      = iv_rank < _LOW_IV
        high_iv     = iv_rank > _HIGH_IV

        if is_choppy and high_iv:
            play = _iron_condor(chain, result)
        elif is_bearish and high_iv:
            play = _bear_call_spread(chain, result)
        elif is_bearish and low_iv:
            play = _long_put(chain, result)
        elif is_bullish and high_iv:
            play = _bull_put_spread(chain, result) or _call_debit_spread(chain, result)
        elif is_bullish and low_iv:
            play = _long_call(chain, result)
        else:
            # Default: cash-secured put if high IV + neutral/bullish
            if high_iv:
                play = _cash_secured_put(chain, result)
            else:
                play = _call_debit_spread(chain, result)

        if play:
            play["upside_score"]  = round(upside, 3)
            play["sector"]        = result.get("sector", "")
            play["narrative"]     = result.get("narrative", {})
            play["confidence"]    = _confidence(result, play["type"], iv_rank)
            play["setups"]        = setups_list
            play["last_price"]    = result.get("last_price", 0)
            plays.append(play)

    # Sort: credit plays first (premium collection), then by confidence
    plays.sort(key=lambda x: (x["type"] == "debit", -x["confidence"]))
    return plays[:15]
