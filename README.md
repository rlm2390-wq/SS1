# 📈 Stock Discovery Bot

A multi-brain stock scanner that identifies equities with asymmetric upside potential. It thinks like a trader and an institutional buyer — analyzing technical structure, fundamentals, sentiment, short squeeze setups, and market regime simultaneously before surfacing any alert.

> **Current state:** Runs on synthetic data out of the box. Plug in a real data provider (yfinance, Polygon, Alpaca) in `data/market_data.py` to go live.

---

## What It Does

Every scan runs the full universe through 9 independent "brains":

| Brain | What it measures |
|---|---|
| **Regime** | Market context — SPY trend, breadth, VIX, put/call ratio |
| **Technical** | Trend alignment, volatility compression, RSI, volume accumulation |
| **Fundamental** | Revenue/EPS growth, valuation vs sector, balance sheet quality |
| **Sentiment** | News sentiment, analyst revisions, options flow |
| **Structural** | Short interest, days to cover, float size, insider buying |
| **Risk** | Liquidity, ATR, gap history, price-level risk |
| **Setups** | Pre-move patterns: breakout, trend pullback, short squeeze, earnings drift |
| **Validation** | Data sanity checks — catches bad data before it poisons scores |
| **Scoring** | Adaptive weighted UpsideScore combining all brains |

An alert only fires when **all five gates pass simultaneously:**
- Market regime is supportive
- UpsideScore clears the threshold
- RiskScore is acceptable
- A setup is detected
- UpsideScore improved vs the previous scan

---

## Project Structure

```
stock_discovery_bot/
├── app.py                  # Flask web server (dashboard + API)
├── main.py                 # CLI scan runner
├── config.py               # All thresholds and weights
├── Procfile                # Railway/Heroku deployment
├── railway.json            # Railway config
├── requirements.txt
│
├── templates/
│   └── dashboard.html      # Dark terminal-style web UI
│
├── data/
│   ├── universe.py         # Ticker list / universe filter
│   └── market_data.py      # Data access layer (swap for real provider here)
│
├── brains/
│   ├── regime.py           # Market context brain
│   ├── technical.py        # Technical analysis brain
│   ├── fundamental.py      # Fundamental analysis brain
│   ├── sentiment.py        # Sentiment + options flow brain
│   ├── structural.py       # Short interest, float, insider brain
│   ├── risk.py             # Risk scoring brain
│   ├── setups.py           # Setup detection brain
│   ├── scoring.py          # UpsideScore engine (adaptive weights)
│   └── validation.py       # Data sanity brain
│
├── storage/
│   └── history.py          # In-memory history store (percentiles, z-scores)
│
└── alerts/
    └── notifier.py         # Alert dispatcher (console, file, Telegram, email)
```

---

## Quickstart

**Requirements:** Python 3.9+

```bash
# 1. Install dependencies
pip install flask numpy gunicorn

# 2. Run the web dashboard
python app.py
# Open http://localhost:5000

# 3. Or run a CLI scan
python main.py
```

---

## Deploy to Railway

1. Push the project to a GitHub repo
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
3. Select your repo — Railway auto-detects `Procfile` and `requirements.txt`
4. Hit **Deploy** — you get a public URL in ~2 minutes

No extra configuration needed. Railway reads `railway.json` automatically.

---

## Configuration

Everything lives in `config.py`.

### Alert thresholds

```python
ALERT_CONFIG = {
    "upside_percentile_threshold": 0.65,  # minimum UpsideScore to alert
    "risk_percentile_max":         0.60,  # maximum RiskScore allowed
    "setup_percentile_threshold":  0.55,  # minimum setup score
    "min_upside_change":           0.05,  # min improvement vs previous scan
    "min_regime_score":            0.40,  # market must be at least this healthy
}
```

Lower `upside_percentile_threshold` and `setup_percentile_threshold` to see more alerts. Raise them to be more selective.

### Scoring weights

```python
SCORING_WEIGHTS = {
    "technical":   0.25,
    "fundamental": 0.25,
    "sentiment":   0.15,
    "structural":  0.15,
    "setup":       0.10,
    "regime":      0.10,
}
```

These are the static fallback weights. When `ADAPTIVE_WEIGHT_ENABLED = True`, the bot shifts weights toward whichever factors have been most predictive recently (rolling correlation vs forward score changes).

### Alert channels

```python
NOTIFICATION_CONFIG = {
    "channel": "console",   # options: "console" | "file" | "telegram" | "email"
    "output_file": "alerts.jsonl",
}
```

---

## Connecting Real Data

All data fetching is isolated in `data/market_data.py`. The three functions to replace are:

```python
def get_index_data()        # SPY/QQQ/IWM, VIX, breadth
def get_stock_data(ticker)  # prices, volume, fundamentals, short interest, news, options
def get_sector_stats(ticker) # sector-median valuation stats
```

**Easiest option — yfinance (free):**

```bash
pip install yfinance
```

```python
# In data/market_data.py
import yfinance as yf

def get_stock_data(ticker, lookback_days=252):
    t = yf.Ticker(ticker)
    hist = t.history(period="1y")
    info = t.info
    return {
        "prices":  hist["Close"].values,
        "volumes": hist["Volume"].values,
        # ... map the rest of the fields
    }
```

**Other supported providers:** Polygon.io, Alpaca Markets, Interactive Brokers, Tradier.

---

## Setup Types

The setup brain detects four pre-move patterns:

| Setup | Trigger conditions |
|---|---|
| `volatility_breakout` | Bollinger Band compression + price near 52-week high + strong trend |
| `trend_pullback` | Uptrend intact + price at MA20/50 support + RSI cooling + declining pullback volume |
| `short_squeeze` | High short float (>15%) + days-to-cover >3 + price turning up + volume accelerating |
| `earnings_drift` | EPS beat >5% + price not yet overbought + strong fundamentals |

Multiple setups on the same ticker stack (with diminishing credit) and increase the composite `setup_score`.

---

## Web Dashboard

The dashboard at `/` updates every 2 seconds and shows:

- **Market regime** badge with color-coded status
- **Stats bar** — tickers scanned, alerts fired, average UpsideScore
- **Alert cards** — detailed breakdown for every ticker that passed all filters
- **Full results table** — all tickers ranked by UpsideScore with mini factor bars

Trigger a new scan manually with the **▶ Run Scan** button.

---

## Extending the Bot

**Add a new brain:**
1. Create `brains/mybrain.py` with `compute_X_factors()` and `score_X_factors()`
2. Call it in `main.py` inside `score_ticker()`
3. Add its score to `factor_scores` dict and update weights in `config.py`

**Add a new setup:**
1. Write a `_detect_mysetup()` function in `brains/setups.py`
2. Call it inside `detect_setups()` and append to `detected_setups`

**Add a new alert channel:**
1. Add a new `elif channel == "myservice":` block in `alerts/notifier.py`

---

## Disclaimer

This tool is for research and educational purposes only. It does not constitute financial advice. Always do your own due diligence before making any investment decision.
