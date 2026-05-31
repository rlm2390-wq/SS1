# ─────────────────────────────────────────────
#  main.py  –  Main scan runner
# ─────────────────────────────────────────────
from __future__ import annotations
import datetime
import sys
from typing import Dict, Any, List

from config import ALERT_CONFIG, HISTORY_CONFIG
from data.universe import get_universe
from data.market_data import get_index_data, get_stock_data, get_sector_stats
from storage.history import HistoryStore
from brains import regime, technical, fundamental, sentiment, structural, risk, setups, scoring, validation
from alerts.notifier import send_alert


# ── Alert filter ──────────────────────────────────────────────────────────────

def should_alert(
    regime_score: float,
    upside: float,
    risk_score: float,
    setup_score: float,
    upside_change: float,
    alert_config: Dict[str, Any],
) -> bool:
    """
    Gate logic for firing an alert.  All five conditions must pass.

    1. Regime must be at least minimally supportive.
    2. UpsideScore must clear the threshold.
    3. RiskScore must be acceptable.
    4. At least one meaningful setup must be present.
    5. UpsideScore must have improved vs previous scan.
    """
    if regime_score < alert_config["min_regime_score"]:
        return False
    if upside < alert_config["upside_percentile_threshold"]:
        return False
    if risk_score > alert_config["risk_percentile_max"]:
        return False
    if setup_score < alert_config["setup_percentile_threshold"]:
        return False
    if upside_change < alert_config["min_upside_change"]:
        return False
    return True


# ── Per-ticker scoring ────────────────────────────────────────────────────────

def score_ticker(
    ticker: str,
    regime_label: str,
    regime_score: float,
    history_store: HistoryStore,
) -> Dict[str, Any] | None:
    """
    Run all brains for a single ticker.  Returns a result dict or None if invalid.
    """
    stock_data  = get_stock_data(ticker, HISTORY_CONFIG["lookback_days"])
    sector_stats = get_sector_stats(ticker)

    # Validation
    is_valid, issues = validation.validate_data(stock_data)
    if not is_valid:
        return None

    # ── Factor brains ─────────────────────────────────────────────────────────
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

    # ── Setups ────────────────────────────────────────────────────────────────
    factor_scores = {
        "technical":   tech_score,
        "fundamental": fund_score,
        "sentiment":   sent_score,
        "structural":  struct_score,
        "risk":        risk_score,
    }
    setups_list, setup_score = setups.detect_setups(
        stock_data, factor_scores, regime_label, history_store
    )

    # ── UpsideScore ───────────────────────────────────────────────────────────
    upside = scoring.compute_upside_score(
        tech_score, fund_score, sent_score, struct_score,
        setup_score, regime_score, history_store,
    )

    # ── Change vs previous run ────────────────────────────────────────────────
    prev_history = history_store.get_stock_history(ticker, "upside", lookback_days=3)
    prev_upside  = float(prev_history[-1]) if prev_history else 0.0
    upside_change = upside - prev_upside

    # ── Persist to history ────────────────────────────────────────────────────
    history_store.update_stock(ticker, date=None, data={
        "upside":    upside,
        "risk":      risk_score,
        "technical": tech_score,
        "fundamental": fund_score,
        "sentiment": sent_score,
        "structural": struct_score,
        "setup":     setup_score,
    })

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
        "issues":       issues,
        "sector":       stock_data.get("sector", "Unknown"),
        "last_price":   float(stock_data["prices"][-1]),
    }


# ── Main scan ─────────────────────────────────────────────────────────────────

def run_scan(verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Full market scan.  Returns the list of alert payloads fired.
    """
    history_store = HistoryStore()
    today = datetime.date.today()

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Stock Discovery Bot  –  {today.isoformat()}")
        print(f"{'='*55}\n")

    # ── Market context ────────────────────────────────────────────────────────
    index_data = get_index_data()
    regime_label, regime_score = regime.compute_market_context(index_data, history_store)

    if verbose:
        print(f"  Market Regime  : {regime_label}")
        print(f"  Regime Score   : {regime_score:.3f}")

    # ── Universe ──────────────────────────────────────────────────────────────
    tickers = get_universe()
    if verbose:
        print(f"  Universe size  : {len(tickers)} tickers\n")

    all_results: List[Dict[str, Any]] = []
    alerts: List[Dict[str, Any]] = []

    for ticker in tickers:
        result = score_ticker(ticker, regime_label, regime_score, history_store)
        if result is None:
            continue

        all_results.append(result)

        if should_alert(
            regime_score,
            result["upside"],
            result["risk"],
            result["setup_score"],
            result["upside_change"],
            ALERT_CONFIG,
        ):
            alerts.append(result)

    # ── Rank and dispatch alerts ───────────────────────────────────────────────
    alerts.sort(key=lambda x: x["upside"], reverse=True)

    if verbose:
        print(f"\n  Scan complete.  {len(all_results)} tickers processed.")
        print(f"  Alerts fired  : {len(alerts)}\n")

    for a in alerts:
        send_alert(a)

    # ── Summary table (verbose) ───────────────────────────────────────────────
    if verbose and not alerts:
        print("  No tickers met all alert thresholds this scan.")
        print("  Top 5 by UpsideScore:\n")
        top5 = sorted(all_results, key=lambda x: x["upside"], reverse=True)[:5]
        print(f"  {'Ticker':<8} {'Upside':>7} {'Risk':>6} {'Setups'}")
        print("  " + "-" * 45)
        for r in top5:
            s = ", ".join(r["setups"]) if r["setups"] else "—"
            print(f"  {r['ticker']:<8} {r['upside']:>7.3f} {r['risk']:>6.3f}  {s}")

    return alerts


if __name__ == "__main__":
    verbose = "--quiet" not in sys.argv
    run_scan(verbose=verbose)
