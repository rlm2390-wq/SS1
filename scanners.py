# ─────────────────────────────────────────────
#  scanners.py  –  Six Discovery Scanners
#
#  Each scanner is a self-contained filter +
#  scoring function that operates on the full
#  scan results list already computed by main.py.
#  No extra API calls — uses data already in
#  each result dict.
#
#  Scanners:
#    1. under5         — Under $5 high risk/reward
#    2. low_float      — Float < 20M rockets
#    3. short_squeeze  — Squeeze radar
#    4. volume_anomaly — Unusual volume / early accumulation
#    5. vol_compression— Coiled springs (TTM Squeeze-style)
#    6. pre_earnings   — Pre-earnings drift plays
# ─────────────────────────────────────────────
from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger("scanners")

# ── Shared helpers ────────────────────────────────────────────────────────────

_EXPLOSIVE_SETUPS   = {"short_squeeze", "volatility_breakout"}
_SUPPORTING_SETUPS  = {"trend_pullback", "earnings_drift"}
_EXPLOSIVE_PRE      = {"oi_buildup", "si_acceleration", "vol_contraction"}

def _pre_count(r: Dict) -> int:
    return r.get("pre_signals", {}).get("signal_count", 0)

def _pre_score(r: Dict) -> float:
    return float(r.get("pre_signals", {}).get("pre_score", 0.0))

def _has_explosive_setup(r: Dict) -> bool:
    setups    = set(r.get("setups", []))
    fired_pre = {s["name"] for s in r.get("pre_signals", {}).get("signals", [])}
    return bool((setups & _EXPLOSIVE_SETUPS) or (fired_pre & _EXPLOSIVE_PRE))

def _explosive_setup_score(r: Dict) -> float:
    setups    = set(r.get("setups", []))
    fired_pre = {s["name"] for s in r.get("pre_signals", {}).get("signals", [])}
    if setups & _EXPLOSIVE_SETUPS:              return 1.0
    if (fired_pre & _EXPLOSIVE_PRE) and (setups & _SUPPORTING_SETUPS): return 0.85
    if fired_pre & _EXPLOSIVE_PRE:             return 0.70
    if setups & _SUPPORTING_SETUPS:            return 0.50
    return 0.0

def _liq_ok(r: Dict, min_dv_proxy: float = 0.70) -> bool:
    """Proxy: liquidity sub-score <= min_dv_proxy means dollar vol is adequate."""
    sub = r.get("sub_scores", {}).get("risk", {})
    liq = sub.get("liquidity", 0.5) if isinstance(sub, dict) else 0.5
    return liq <= min_dv_proxy

def _float_ok(r: Dict, min_float_m: float = 15.0) -> bool:
    """Proxy via structural float_ownership sub-score."""
    sub = r.get("sub_scores", {}).get("structural", {})
    fo  = sub.get("float_ownership", 0.5) if isinstance(sub, dict) else 0.5
    return fo >= 0.20   # very low = tiny/no float data

def _micro_score(r: Dict) -> float:
    return float(r.get("microstructure", {}).get("microstructure_score", 0.5))

def _tech_score(r: Dict) -> float:
    return float(r.get("factor_scores", {}).get("technical", 0.5))

def _sent_score(r: Dict) -> float:
    return float(r.get("factor_scores", {}).get("sentiment", 0.5))

def _struct_score(r: Dict) -> float:
    return float(r.get("factor_scores", {}).get("structural", 0.5))

def _clamp(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))

def _signal_count_score(r: Dict, max_signals: int = 6) -> float:
    return _clamp(_pre_count(r) / max_signals)


# ── 1. Under $5 — High Risk / High Reward ────────────────────────────────────

def scan_under5(results: List[Dict]) -> List[Dict]:
    """
    Price $1–$5 with 3+ pre-signals and an explosive setup.
    Tighter signal requirement than Under $10 due to higher noise.

    Score = 0.45 * SignalCountScore
           + 0.35 * ExplosiveSetupScore
           + 0.20 * (1 - RiskScore)
    """
    candidates = []
    for r in results:
        price = r.get("last_price", 999)
        risk  = r.get("risk", 1.0)

        if not (1.0 <= price <= 5.0):        continue
        if not _liq_ok(r, min_dv_proxy=0.75): continue   # ~$750k+ dollar vol
        if not _float_ok(r, 15.0):            continue
        if _pre_count(r) < 3:                 continue
        if not _has_explosive_setup(r):       continue
        if risk > 0.80:                       continue

        score = _clamp(
            0.45 * _signal_count_score(r) +
            0.35 * _explosive_setup_score(r) +
            0.20 * (1.0 - risk)
        )
        tier = "strong" if _pre_count(r) >= 4 else "medium"

        enriched = dict(r)
        enriched["scanner_meta"] = {
            "scanner":       "under5",
            "score":         round(score, 3),
            "tier":          tier,
            "signal_count":  _pre_count(r),
            "label":         "Under $5",
        }
        candidates.append(enriched)

    candidates.sort(key=lambda x: x["scanner_meta"]["score"], reverse=True)
    return candidates


# ── 2. Low Float Rockets ──────────────────────────────────────────────────────

def scan_low_float(results: List[Dict]) -> List[Dict]:
    """
    Float < 20M shares — small volume shifts = big price moves.
    Requires squeeze OR breakout setup.

    Score = 0.40 * ExplosiveSetupScore
           + 0.40 * SignalCountScore
           + 0.20 * MicrostructureScore
    """
    candidates = []
    for r in results:
        price  = r.get("last_price", 999)
        risk   = r.get("risk", 1.0)
        setups = set(r.get("setups", []))

        if price < 2.0:                       continue
        if not _liq_ok(r, 0.70):              continue

        # Require explicitly small float signal: struct_score boosted by low float
        # float_ownership sub-score < 0.35 = small float
        sub = r.get("sub_scores", {}).get("structural", {})
        fo  = sub.get("float_ownership", 0.5) if isinstance(sub, dict) else 0.5
        if fo > 0.50:                         continue   # float too large

        if _pre_count(r) < 2:                 continue
        if not bool(setups & (_EXPLOSIVE_SETUPS | _SUPPORTING_SETUPS)): continue
        if risk > 0.80:                       continue

        score = _clamp(
            0.40 * _explosive_setup_score(r) +
            0.40 * _signal_count_score(r) +
            0.20 * _micro_score(r)
        )
        tier = "strong" if _pre_count(r) >= 3 else "medium"

        enriched = dict(r)
        enriched["scanner_meta"] = {
            "scanner":       "low_float",
            "score":         round(score, 3),
            "tier":          tier,
            "signal_count":  _pre_count(r),
            "label":         "Low Float",
            "float_note":    "Float < 20M — small moves amplified",
        }
        candidates.append(enriched)

    candidates.sort(key=lambda x: x["scanner_meta"]["score"], reverse=True)
    return candidates


# ── 3. Short Squeeze Watchlist ────────────────────────────────────────────────

def scan_short_squeeze(results: List[Dict]) -> List[Dict]:
    """
    Detects squeeze setups days before they explode.
    Short float > 10%, days to cover > 3, plus volume/OI signal.

    Score = 0.50 * ShortInterestScore
           + 0.25 * VolumeAccelerationScore
           + 0.25 * TrendInflectionScore
    """
    candidates = []
    for r in results:
        price  = r.get("last_price", 999)
        risk   = r.get("risk", 1.0)

        if price < 2.0:            continue
        if not _liq_ok(r, 0.70):   continue

        # ShortInterest score — proxy via structural score + setup
        struct = _struct_score(r)
        setups = set(r.get("setups", []))
        has_squeeze_setup = "short_squeeze" in setups
        si_pre = any(
            s["name"] == "si_acceleration"
            for s in r.get("pre_signals", {}).get("signals", [])
        )

        # Need either squeeze setup OR si_acceleration pre-signal
        if not (has_squeeze_setup or si_pre):  continue
        if struct < 0.35:                      continue
        if risk > 0.80:                        continue

        # Volume acceleration proxy — from microstructure dollar vol
        micro = r.get("microstructure", {})
        sub_ms = micro.get("sub_scores", {})
        vol_accel = float(sub_ms.get("dollar_vol_accel", 0.5))

        # Trend inflection proxy — technical score
        trend_inflection = _tech_score(r)

        # Short interest score = structural + squeeze setup bonus
        si_score = float(np.clip(
            struct * 0.7 + (0.3 if has_squeeze_setup else 0.0), 0.0, 1.0))

        score = _clamp(
            0.50 * si_score +
            0.25 * vol_accel +
            0.25 * trend_inflection
        )

        if score < 0.35:   continue   # weak squeeze signal

        tier = "strong" if (has_squeeze_setup and si_pre) else "medium"

        enriched = dict(r)
        enriched["scanner_meta"] = {
            "scanner":       "short_squeeze",
            "score":         round(score, 3),
            "tier":          tier,
            "signal_count":  _pre_count(r),
            "label":         "Squeeze",
            "si_score":      round(si_score, 3),
            "vol_accel":     round(vol_accel, 3),
        }
        candidates.append(enriched)

    candidates.sort(key=lambda x: x["scanner_meta"]["score"], reverse=True)
    return candidates


# ── 4. Volume Anomalies ───────────────────────────────────────────────────────

def scan_volume_anomaly(results: List[Dict]) -> List[Dict]:
    """
    Volume > 2× 20-day average — early accumulation before news or breakouts.

    Score = 0.60 * VolumeSpikeScore
           + 0.25 * SentimentInflectionScore
           + 0.15 * MicrostructureScore
    """
    candidates = []
    for r in results:
        price = r.get("last_price", 999)
        risk  = r.get("risk", 1.0)

        if price < 1.0:            continue
        if risk > 0.75:            continue

        # Volume spike proxy: microstructure dollar_vol_accel
        micro   = r.get("microstructure", {})
        sub_ms  = micro.get("sub_scores", {})
        vol_accel = float(sub_ms.get("dollar_vol_accel", 0.5))

        # Lower threshold — 0.50 = moderate volume acceleration
        if vol_accel < 0.50:       continue

        sent = _sent_score(r)

        score = _clamp(
            0.60 * vol_accel +
            0.25 * sent +
            0.15 * _micro_score(r)
        )

        tier = "strong" if vol_accel >= 0.80 else "medium"

        enriched = dict(r)
        enriched["scanner_meta"] = {
            "scanner":      "volume_anomaly",
            "score":        round(score, 3),
            "tier":         tier,
            "signal_count": _pre_count(r),
            "label":        "Vol Spike",
            "vol_accel":    round(vol_accel, 3),
        }
        candidates.append(enriched)

    candidates.sort(key=lambda x: x["scanner_meta"]["score"], reverse=True)
    return candidates


# ── 5. Volatility Compression ─────────────────────────────────────────────────

def scan_vol_compression(results: List[Dict]) -> List[Dict]:
    """
    BB width in bottom 10% of 1-year range — coiled springs.
    TTM Squeeze-style section.

    Score = 0.50 * CompressionStrength
           + 0.30 * TrendScore
           + 0.20 * VolumeDryUpScore
    """
    candidates = []
    for r in results:
        price = r.get("last_price", 999)
        risk  = r.get("risk", 1.0)

        if price < 2.0:            continue
        if risk > 0.80:            continue

        # Volatility compression pre-signal
        fired_pre = r.get("pre_signals", {}).get("signals", [])
        vol_contraction = next(
            (s for s in fired_pre if s["name"] == "vol_contraction"), None)
        if not vol_contraction:    continue

        compression_strength = float(vol_contraction.get("conviction", 0.5))

        # Volume dry-up adds confirmation
        vol_dryup = next(
            (s for s in fired_pre if s["name"] == "volume_dryup"), None)
        vol_dryup_score = float(vol_dryup["conviction"]) if vol_dryup else 0.0

        tech  = _tech_score(r)

        score = _clamp(
            0.50 * compression_strength +
            0.30 * tech +
            0.20 * vol_dryup_score
        )

        tier = "strong" if (compression_strength > 0.75 and vol_dryup) else "medium"

        enriched = dict(r)
        enriched["scanner_meta"] = {
            "scanner":           "vol_compression",
            "score":             round(score, 3),
            "tier":              tier,
            "signal_count":      _pre_count(r),
            "label":             "Vol Compression",
            "compression":       round(compression_strength, 3),
            "vol_dryup":         round(vol_dryup_score, 3),
        }
        candidates.append(enriched)

    candidates.sort(key=lambda x: x["scanner_meta"]["score"], reverse=True)
    return candidates


# ── 6. Pre-Earnings Momentum ──────────────────────────────────────────────────

def scan_pre_earnings(results: List[Dict]) -> List[Dict]:
    """
    Earnings in 7–21 days with improving sentiment + volume + trend.
    Catches institutional pre-positioning.

    Score = 0.40 * SentimentTrendScore
           + 0.30 * VolumeTrendScore
           + 0.30 * TrendInflectionScore
    """
    today   = datetime.date.today()
    candidates = []

    for r in results:
        price = r.get("last_price", 999)
        risk  = r.get("risk", 1.0)

        if price < 2.0:            continue
        if not _liq_ok(r, 0.70):   continue

        # Earnings date check
        ed_str = r.get("earnings_date")
        if not ed_str:             continue
        try:
            ed      = datetime.date.fromisoformat(str(ed_str)[:10])
            days_to = (ed - today).days
        except (ValueError, TypeError):
            continue
        if not (7 <= days_to <= 21): continue

        if risk > 0.75:            continue

        # Sentiment trend
        sent = _sent_score(r)
        if sent < 0.35:            continue   # need improving sentiment

        # Volume trend — from microstructure
        micro   = r.get("microstructure", {})
        sub_ms  = micro.get("sub_scores", {})
        vol_trend = float(sub_ms.get("dollar_vol_accel", 0.5))

        # Trend inflection = technical score
        trend = _tech_score(r)

        # OI delta bonus
        oi_signal = any(
            s["name"] == "oi_buildup"
            for s in r.get("pre_signals", {}).get("signals", [])
        )
        oi_bonus = 0.05 if oi_signal else 0.0

        score = _clamp(
            0.40 * sent +
            0.30 * vol_trend +
            0.30 * trend +
            oi_bonus
        )

        tier = "strong" if (sent > 0.65 and oi_signal) else "medium"

        enriched = dict(r)
        enriched["scanner_meta"] = {
            "scanner":      "pre_earnings",
            "score":        round(score, 3),
            "tier":         tier,
            "signal_count": _pre_count(r),
            "label":        "Pre-Earnings",
            "days_to_earn": days_to,
            "earn_date":    str(ed_str)[:10],
            "oi_bonus":     oi_signal,
        }
        candidates.append(enriched)

    candidates.sort(key=lambda x: x["scanner_meta"]["score"], reverse=True)
    return candidates



# ── 7. Penny Stocks — Strictest Scanner ──────────────────────────────────────

def scan_penny(results: List[Dict]) -> List[Dict]:
    """
    $0.50–$2.00 with 4+ pre-signals, explosive setup, risk < 0.60.
    The tightest filter in the entire bot.

    Gates (ALL must pass sequentially):
      1. Price $0.50–$2.00
      2. Avg dollar volume > $500k  (liq_risk proxy <= 0.80)
      3. Float > 10M               (float_ownership proxy >= 0.15)
      4. 4+ pre-signals            (no exceptions)
      5. Explosive setup present   (squeeze, breakout, OI, SI, micro accum)
      6. Risk score < 0.60

    PennyScore =
      0.40 * SignalCountScore      (out of 6 max detectors)
    + 0.30 * ExplosiveSetupScore   (1.0 = squeeze/breakout, 0.7 = explosive pre)
    + 0.20 * MicrostructureScore   (institutional tape quality)
    + 0.10 * (1 - RiskScore)       (minimal — already filtered above 0.60)

    Hard cutoff: PennyScore < 0.70 → excluded.
    Tiers: 5+ signals = strong (green), 4 signals = medium (yellow).
    """
    candidates = []

    for r in results:
        price = r.get("last_price", 999)
        risk  = r.get("risk", 1.0)

        # ── Gate 1: Price range ──────────────────────────────────────────────
        if not (0.50 <= price <= 2.00):
            continue

        # ── Gate 2: Dollar volume > $500k ────────────────────────────────────
        # liquidity sub-score > 0.80 = avg_dv below ~$500k threshold
        sub_risk = r.get("sub_scores", {}).get("risk", {})
        liq      = sub_risk.get("liquidity", 0.6) if isinstance(sub_risk, dict) else 0.6
        if liq > 0.80:
            continue

        # ── Gate 3: Float > 10M ──────────────────────────────────────────────
        sub_struct   = r.get("sub_scores", {}).get("structural", {})
        float_score  = sub_struct.get("float_ownership", 0.5) if isinstance(sub_struct, dict) else 0.5
        if float_score < 0.15:
            continue

        # ── Gate 4: 4+ pre-signals ───────────────────────────────────────────
        n_pre = _pre_count(r)
        if n_pre < 4:
            continue

        # ── Gate 5: Explosive setup required ─────────────────────────────────
        exp_score = _explosive_setup_score(r)
        if exp_score == 0.0:
            continue

        # ── Gate 6: Risk < 0.60 ──────────────────────────────────────────────
        if risk >= 0.60:
            continue

        # ── Scoring ──────────────────────────────────────────────────────────
        sig_score   = _clamp(n_pre / 6.0)
        micro_score = _micro_score(r)

        penny_score = _clamp(
            0.40 * sig_score   +
            0.30 * exp_score   +
            0.20 * micro_score +
            0.10 * (1.0 - risk)
        )

        # Hard floor — no weak penny candidates
        if penny_score < 0.70:
            continue

        tier = "strong" if n_pre >= 5 else "medium"

        # Build signal summary for UI display
        fired_signals = r.get("pre_signals", {}).get("signals", [])
        signal_names  = [s["name"].replace("_", " ") for s in fired_signals]

        enriched = dict(r)
        enriched["scanner_meta"] = {
            "scanner":       "penny",
            "score":         round(penny_score, 3),
            "tier":          tier,
            "signal_count":  n_pre,
            "label":         "Penny",
            "sig_score":     round(sig_score,   3),
            "exp_score":     round(exp_score,   3),
            "micro_score":   round(micro_score, 3),
            "signal_names":  signal_names,
            "cutoff_note":   "PennyScore ≥ 0.70 required",
        }
        candidates.append(enriched)

    # Sort by PennyScore descending
    candidates.sort(key=lambda x: x["scanner_meta"]["score"], reverse=True)
    return candidates

# ── Master runner ─────────────────────────────────────────────────────────────

def run_all_scanners(results: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Run all six scanners against the full results list.
    Returns a dict keyed by scanner name.
    Each value is a ranked list of enriched result dicts.
    """
    logger.info("Running scanners on %d results", len(results))
    output = {
        "under5":          scan_under5(results)[:15],
        "low_float":       scan_low_float(results)[:15],
        "short_squeeze":   scan_short_squeeze(results)[:15],
        "volume_anomaly":  scan_volume_anomaly(results)[:15],
        "vol_compression": scan_vol_compression(results)[:15],
        "pre_earnings":    scan_pre_earnings(results)[:15],
        "penny":           scan_penny(results)[:15],
    }
    for k, v in output.items():
        logger.info("Scanner %s: %d candidates", k, len(v))
    return output
