# ─────────────────────────────────────────────
#  positions.py  –  Position Tracker
#
#  Server-side JSON storage for open/closed
#  positions. Survives Railway restarts.
#  Live P&L computed against current scan prices.
#
#  Fields:
#    id, ticker, entry, stop, target1, target2,
#    shares, setup_type, date_opened, notes,
#    status (open/closed), exit_price, exit_date
# ─────────────────────────────────────────────
from __future__ import annotations

import datetime
import json
import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger("positions")

_POSITIONS_FILE = "positions.json"
_lock           = threading.Lock()


# ── Storage ───────────────────────────────────────────────────────────────────

def _load() -> List[Dict]:
    if not os.path.exists(_POSITIONS_FILE):
        return []
    try:
        with open(_POSITIONS_FILE) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load positions: %s", e)
        return []


def _save(positions: List[Dict]):
    try:
        with open(_POSITIONS_FILE, "w") as f:
            json.dump(positions, f, indent=2)
    except Exception as e:
        logger.warning("Could not save positions: %s", e)


# ── CRUD ──────────────────────────────────────────────────────────────────────

def get_all_positions() -> List[Dict]:
    with _lock:
        return _load()


def get_open_positions() -> List[Dict]:
    return [p for p in get_all_positions() if p.get("status") == "open"]


def add_position(data: Dict[str, Any]) -> Dict:
    """
    Add a new position. Required: ticker, entry, stop, target1, shares.
    Optional: target2, setup_type, notes.
    """
    pos = {
        "id":          str(uuid.uuid4())[:8],
        "ticker":      data.get("ticker", "").upper().strip(),
        "entry":       float(data.get("entry",   0)),
        "stop":        float(data.get("stop",    0)),
        "target1":     float(data.get("target1", 0)),
        "target2":     float(data.get("target2", 0)),
        "shares":      int(data.get("shares",    0)),
        "setup_type":  data.get("setup_type", ""),
        "notes":       data.get("notes",      ""),
        "date_opened": datetime.date.today().isoformat(),
        "status":      "open",
        "exit_price":  None,
        "exit_date":   None,
        "exit_reason": None,
    }
    if not pos["ticker"] or pos["entry"] <= 0:
        raise ValueError("ticker and entry are required")

    with _lock:
        positions = _load()
        positions.append(pos)
        _save(positions)

    logger.info("Position added: %s @ $%.2f", pos["ticker"], pos["entry"])
    return pos


def close_position(pos_id: str, exit_price: float, exit_reason: str = "manual") -> Optional[Dict]:
    """Close a position by ID. Returns the updated position or None."""
    with _lock:
        positions = _load()
        for p in positions:
            if p["id"] == pos_id and p["status"] == "open":
                p["status"]     = "closed"
                p["exit_price"] = round(float(exit_price), 2)
                p["exit_date"]  = datetime.date.today().isoformat()
                p["exit_reason"]= exit_reason
                _save(positions)
                logger.info("Position closed: %s @ $%.2f (%s)", p["ticker"], exit_price, exit_reason)
                return p
    return None


def update_position(pos_id: str, data: Dict) -> Optional[Dict]:
    """Update notes, stop, or targets on an open position."""
    with _lock:
        positions = _load()
        for p in positions:
            if p["id"] == pos_id:
                for field in ("stop", "target1", "target2", "notes", "shares"):
                    if field in data:
                        val = data[field]
                        p[field] = float(val) if field != "notes" else str(val)
                        if field == "shares":
                            p[field] = int(val)
                _save(positions)
                return p
    return None


def delete_position(pos_id: str) -> bool:
    """Hard delete — removes from storage entirely."""
    with _lock:
        positions = _load()
        before    = len(positions)
        positions = [p for p in positions if p["id"] != pos_id]
        if len(positions) < before:
            _save(positions)
            return True
    return False


# ── Live P&L enrichment ───────────────────────────────────────────────────────

def enrich_with_pnl(positions: List[Dict], price_map: Dict[str, float]) -> List[Dict]:
    """
    Attach live P&L and status flags to each open position.
    price_map: {ticker: current_price}
    """
    enriched = []
    for p in positions:
        pos    = dict(p)
        ticker = pos["ticker"]
        entry  = pos["entry"]
        stop   = pos["stop"]
        t1     = pos["target1"]
        t2     = pos["target2"]
        shares = pos["shares"]

        if pos["status"] == "open" and ticker in price_map:
            current  = price_map[ticker]
            pnl_sh   = round(current - entry, 2)
            pnl_tot  = round(pnl_sh * shares, 2)
            pnl_pct  = round(pnl_sh / entry * 100, 2) if entry else 0
            risk_sh  = round(entry - stop, 2)
            r_mult   = round(pnl_sh / risk_sh, 2) if risk_sh > 0 else 0

            # Status flags
            at_risk    = current <= stop * 1.01       # within 1% of stop
            hit_t1     = current >= t1
            hit_t2     = current >= t2 if t2 else False
            pct_to_t1  = round((t1 - current) / current * 100, 1) if t1 else None
            pct_to_stop= round((current - stop) / current * 100, 1)

            pos["current_price"] = round(current, 2)
            pos["pnl_per_share"] = pnl_sh
            pos["pnl_total"]     = pnl_tot
            pos["pnl_pct"]       = pnl_pct
            pos["r_multiple"]    = r_mult
            pos["at_risk"]       = at_risk
            pos["hit_t1"]        = hit_t1
            pos["hit_t2"]        = hit_t2
            pos["pct_to_t1"]     = pct_to_t1
            pos["pct_to_stop"]   = pct_to_stop

        elif pos["status"] == "closed" and pos.get("exit_price"):
            ep       = pos["exit_price"]
            pnl_sh   = round(ep - entry, 2)
            pnl_tot  = round(pnl_sh * shares, 2)
            pnl_pct  = round(pnl_sh / entry * 100, 2) if entry else 0
            risk_sh  = round(entry - stop, 2)
            r_mult   = round(pnl_sh / risk_sh, 2) if risk_sh > 0 else 0
            pos["pnl_per_share"] = pnl_sh
            pos["pnl_total"]     = pnl_tot
            pos["pnl_pct"]       = pnl_pct
            pos["r_multiple"]    = r_mult

        enriched.append(pos)

    return enriched


# ── Portfolio summary ─────────────────────────────────────────────────────────

def get_portfolio_summary(positions: List[Dict]) -> Dict:
    """Compute aggregate stats across all enriched open positions."""
    open_pos   = [p for p in positions if p.get("status") == "open"]
    closed_pos = [p for p in positions if p.get("status") == "closed"]

    total_value  = sum(p.get("current_price", p["entry"]) * p["shares"] for p in open_pos)
    total_risk   = sum((p["entry"] - p["stop"]) * p["shares"] for p in open_pos if p["stop"])
    total_pnl    = sum(p.get("pnl_total", 0) for p in open_pos)
    at_risk_cnt  = sum(1 for p in open_pos if p.get("at_risk"))
    t1_cnt       = sum(1 for p in open_pos if p.get("hit_t1"))

    # Closed P&L stats
    closed_pnls  = [p.get("pnl_total", 0) for p in closed_pos]
    wins         = [x for x in closed_pnls if x > 0]
    losses       = [x for x in closed_pnls if x <= 0]
    win_rate     = round(len(wins) / len(closed_pnls) * 100, 1) if closed_pnls else None
    avg_win      = round(sum(wins)   / len(wins),   2) if wins   else 0
    avg_loss     = round(sum(losses) / len(losses), 2) if losses else 0

    # Sector concentration
    sectors = {}
    for p in open_pos:
        s = p.get("setup_type", "unknown")
        sectors[s] = sectors.get(s, 0) + 1
    top_concentration = max(sectors.values()) / len(open_pos) if open_pos else 0

    return {
        "open_count":        len(open_pos),
        "closed_count":      len(closed_pos),
        "total_value":       round(total_value, 2),
        "total_risk":        round(total_risk,  2),
        "unrealized_pnl":    round(total_pnl,   2),
        "at_risk_count":     at_risk_cnt,
        "hit_t1_count":      t1_cnt,
        "win_rate":          win_rate,
        "avg_win":           avg_win,
        "avg_loss":          avg_loss,
        "closed_total_pnl":  round(sum(closed_pnls), 2),
        "top_concentration": round(top_concentration * 100, 1),
    }
