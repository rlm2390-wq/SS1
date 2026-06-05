# ─────────────────────────────────────────────
#  main.py  –  Main scan runner
# ─────────────────────────────────────────────
from __future__ import annotations
import datetime
import sys
from typing import Dict, Any, List

from config import ALERT_CONFIG, HISTORY_CONFIG
from universe import get_universe
from market_data import get_index_data, get_stock_data, get_sector_stats
from history import HistoryStore, get_global_store
import regime, technical, fundamental, sentiment, structural, risk, setups, scoring, validation
from notifier import send_alert


# ── Narrative builder ─────────────────────────────────────────────────────────

def build_narrative(ticker, factor_scores, setups_list, risk_score,
                    upside, stock_data, regime_label):
    why, watch_for, risk_flags = [], [], []

    tech   = factor_scores.get("technical",   0)
    fund   = factor_scores.get("fundamental", 0)
    sent   = factor_scores.get("sentiment",   0)
    struct = factor_scores.get("structural",  0)

    prices  = stock_data.get("prices", [])
    rsi     = float(stock_data.get("rsi", 50))
    si      = stock_data.get("short_interest", {})
    sf      = float(si.get("short_float_pct", 0))
    dtc     = float(si.get("days_to_cover",   0))
    f       = stock_data.get("fundamentals",  {})
    rev_yoy = float(f.get("revenue_yoy", 0))
    eps_yoy = float(f.get("eps_yoy",     0))
    options = stock_data.get("options_flow",  {})
    ins     = stock_data.get("insider_activity", {})
    net_buy = float(ins.get("net_buy_usd_90d", 0))
    price   = float(prices[-1]) if len(prices) else 0
    beta    = float(stock_data.get("beta", 1.0))

    if tech >= 0.70:
        ma50  = stock_data.get("ma50", prices)
        above = (prices[-1] > ma50[-1]) if (hasattr(ma50, "__len__") and len(ma50)) else False
        why.append(f"Strong technical setup — price {'above' if above else 'near'} key moving averages with bullish momentum")
    elif tech >= 0.55:
        why.append("Decent technical structure with moderate upside momentum")

    if fund >= 0.70:
        parts = []
        if rev_yoy > 0.15: parts.append(f"revenue growing {round(rev_yoy*100)}% YoY")
        if eps_yoy > 0.15: parts.append(f"earnings up {round(eps_yoy*100)}% YoY")
        why.append(f"Strong fundamentals — {' and '.join(parts)}" if parts else "Strong fundamentals relative to sector peers")
    elif fund >= 0.55:
        why.append("Above-average fundamentals for its sector")

    if sf > 20:
        why.append(f"High short interest ({round(sf)}% of float, {round(dtc,1)} days to cover) — squeeze potential")
    elif sf > 12:
        why.append(f"Elevated short interest ({round(sf)}% of float) — shorts add fuel to any rally")

    if net_buy > 250_000:
        why.append(f"Insider buying — net ${int(net_buy/1000)}k purchased in the last 90 days")

    if options.get("unusual_options"):
        cvr = float(options.get("call_vol_ratio", 1))
        why.append(f"Unusual options activity — call volume {round(cvr,1)}x above average")

    if sent >= 0.65:
        why.append("Positive news sentiment and/or analyst upgrades recently")

    if "volatility_breakout" in setups_list:
        why.append("Price compressing near highs — breakout could be imminent")
    if "trend_pullback" in setups_list:
        why.append("Healthy pullback to support in an uptrend — classic buy-the-dip setup")
    if "short_squeeze" in setups_list:
        why.append("Short squeeze setup — heavy short interest with price starting to turn up")
    if "earnings_drift" in setups_list:
        why.append("Post-earnings drift — beat estimates and hasn't fully priced in the move yet")

    if "volatility_breakout" in setups_list:
        watch_for.append("Volume spike + close above resistance to confirm the breakout")
    if "trend_pullback" in setups_list:
        watch_for.append("Price holding MA support on low volume — look for a reversal candle")
    if "short_squeeze" in setups_list:
        watch_for.append("Sustained volume above 20-day avg and shorts unable to push lower")
    if "earnings_drift" in setups_list:
        watch_for.append("Continued institutional accumulation in weeks after the earnings beat")
    if not watch_for:
        if tech >= 0.60:
            watch_for.append("Confirm momentum holds above key MAs on above-average volume")
        else:
            watch_for.append("Wait for a clearer technical signal before acting")

    if risk_score > 0.50:
        risk_flags.append("Elevated volatility or low liquidity — size this position carefully")
    if rsi > 72:
        risk_flags.append(f"RSI at {round(rsi)} — short-term overbought, consider waiting")
    if price < 5:
        risk_flags.append("Sub-$5 stock — wider spreads, higher volatility")
    if struct <= 0.30:
        risk_flags.append("Weak structural profile — limited insider conviction")
    if fund <= 0.35:
        risk_flags.append("Weak fundamentals — this is a momentum play, not a value story")
    if regime_label in ("risk_off", "panic"):
        risk_flags.append(f"Market regime is {regime_label.replace('_',' ')} — broad headwinds")
    if beta > 1.8:
        risk_flags.append(f"High beta ({beta:.1f}x) — moves amplify market swings significantly")

    if not why:
        why.append("Borderline signal — scores above threshold but conviction is moderate")

    return {
        "why":        why[:4],
        "watch_for":  watch_for[:3],
        "risk_flags": risk_flags[:3],
    }


# ── Position sizing ───────────────────────────────────────────────────────────

def compute_position_size(price: float, risk_score: float,
                           account_size: float = 25_000,
                           risk_pct: float = 0.01) -> Dict:
    """Suggest position size based on account and risk score."""
    dollar_risk  = account_size * risk_pct
    # Adjust for stock's risk: higher risk → smaller position
    adjusted_risk = dollar_risk * (1 - risk_score * 0.5)
    stop_distance = price * 0.07   # assume 7% stop loss
    shares        = int(adjusted_risk / max(stop_distance, 0.01))
    dollar_value  = round(shares * price, 2)
    pct_of_acct   = round(dollar_value / account_size * 100, 1)

    return {
        "suggested_shares":  shares,
        "dollar_value":      dollar_value,
        "pct_of_account":    pct_of_acct,
        "dollar_at_risk":    round(adjusted_risk, 2),
        "stop_price":        round(price * 0.93, 2),
    }


# ── Alert filter ──────────────────────────────────────────────────────────────

def should_alert(regime_score, upside, risk_score, setup_score,
                 upside_change, alert_config):
    if regime_score   < alert_config["min_regime_score"]:            return False
    if upside         < alert_config["upside_percentile_threshold"]: return False
    if risk_score     > alert_config["risk_percentile_max"]:         return False
    if setup_score    < alert_config["setup_percentile_threshold"]:  return False
    if upside_change  < alert_config["min_upside_change"]:           return False
    return True


def is_under20_popper(result):
    p = result.get("last_price", 999)
    if p > 20 or p < 0.50:       return False
    if result["upside"] < 0.45:  return False
    if result["risk"]   > 0.75:  return False
    if result["factor_scores"].get("technical", 0) < 0.45: return False
    return True


# ── Per-ticker scoring ────────────────────────────────────────────────────────

def score_ticker(ticker, regime_label, regime_score, history_store,
                 prev_alert_set=None):
    stock_data   = get_stock_data(ticker, HISTORY_CONFIG["lookback_days"])
    sector_stats = get_sector_stats(ticker)

    is_valid, issues = validation.validate_data(stock_data)
    if not is_valid:
        return None

    raw_tech   = technical.compute_technical_factors(stock_data, history_store)
    tech_score, tech_sub = technical.score_technical_factors(raw_tech, history_store)

    raw_fund   = fundamental.compute_fundamental_factors(stock_data, sector_stats, history_store)
    fund_score, fund_sub = fundamental.score_fundamental_factors(raw_fund, history_store)

    raw_sent   = sentiment.compute_sentiment_factors(stock_data, history_store)
    sent_score, sent_sub = sentiment.score_sentiment_factors(raw_sent, history_store)

    raw_struct = structural.compute_structural_factors(stock_data, history_store)
    struct_score, struct_sub = structural.score_structural_factors(raw_struct, history_store)

    raw_risk   = risk.compute_risk_factors(stock_data, history_store)
    risk_score, risk_sub = risk.score_risk_factors(raw_risk, history_store)

    factor_scores = {
        "technical":   tech_score,
        "fundamental": fund_score,
        "sentiment":   sent_score,
        "structural":  struct_score,
        "risk":        risk_score,
    }

    setups_list, setup_score = setups.detect_setups(
        stock_data, factor_scores, regime_label, history_store)

    upside = scoring.compute_upside_score(
        tech_score, fund_score, sent_score, struct_score,
        setup_score, regime_score, history_store)

    prev_history  = history_store.get_stock_history(ticker, "upside", lookback_days=3)
    prev_upside   = float(prev_history[-1]) if prev_history else 0.0
    upside_change = upside - prev_upside

    history_store.update_stock(ticker, date=None, data={
        "upside": upside, "risk": risk_score,
        "technical": tech_score, "fundamental": fund_score,
        "sentiment": sent_score, "structural": struct_score,
        "setup": setup_score,
    })

    last_price = float(stock_data["prices"][-1])
    beta       = float(stock_data.get("beta", 1.0))

    narrative = build_narrative(ticker, factor_scores, setups_list,
                                risk_score, upside, stock_data, regime_label)

    # "NEW" badge — first time this ticker has fired an alert
    is_new = (prev_alert_set is not None and ticker not in prev_alert_set
              and upside >= ALERT_CONFIG["upside_percentile_threshold"])

    position = compute_position_size(last_price, risk_score)

    sparkline = history_store.get_score_sparkline(ticker, lookback_days=30)

    return {
        "ticker":        ticker,
        "upside":        upside,
        "upside_change": upside_change,
        "risk":          risk_score,
        "regime":        regime_label,
        "setups":        setups_list,
        "setup_score":   setup_score,
        "factor_scores": factor_scores,
        "sub_scores": {
            "technical":   tech_sub,
            "fundamental": fund_sub,
            "sentiment":   sent_sub,
            "structural":  struct_sub,
            "risk":        risk_sub,
        },
        "narrative":    narrative,
        "issues":       issues,
        "sector":       stock_data.get("sector", "Unknown"),
        "last_price":   last_price,
        "beta":         beta,
        "earnings_date": stock_data.get("earnings_date"),
        "pre_market_chg":  stock_data.get("pre_market_chg",  0.0),
        "post_market_chg": stock_data.get("post_market_chg", 0.0),
        "is_new":       is_new,
        "position_size": position,
        "sparkline":    sparkline,
        "52w_high":     stock_data.get("52w_high", 0),
        "52w_low":      stock_data.get("52w_low",  0),
        "analyst_target": stock_data.get("analyst_target_price", 0),
        "market_cap":   stock_data.get("market_cap", 0),
    }


# ── Main scan ─────────────────────────────────────────────────────────────────

def run_scan(verbose=True, mode=None, watchlist=None):
    history_store = get_global_store()

    if verbose:
        print(f"\n{'='*55}\n  Stock Discovery Bot\n{'='*55}\n")

    index_data = get_index_data()
    regime_label, regime_score = regime.compute_market_context(index_data, history_store)

    if verbose:
        print(f"  Regime: {regime_label} ({regime_score:.3f})")

    tickers = get_universe(mode=mode, watchlist=watchlist)
    if verbose:
        print(f"  Universe: {len(tickers)} tickers\n")

    all_results, alerts = [], []

    for ticker in tickers:
        result = score_ticker(ticker, regime_label, regime_score, history_store)
        if result is None:
            continue
        all_results.append(result)
        if should_alert(regime_score, result["upside"], result["risk"],
                        result["setup_score"], result["upside_change"], ALERT_CONFIG):
            alerts.append(result)
            history_store.log_alert(result)

    alerts.sort(key=lambda x: x["upside"], reverse=True)
    history_store.save()

    if verbose:
        print(f"\n  Done. {len(all_results)} tickers, {len(alerts)} alerts.\n")

    for a in alerts:
        send_alert(a)

    return alerts


if __name__ == "__main__":
    run_scan(verbose="--quiet" not in sys.argv)
