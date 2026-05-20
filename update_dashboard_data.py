"""
================================================================================
    update_dashboard_data.py (v6 - Stat-Arb Production Release)
================================================================================
"""
import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
BIST30_TICKERS = [
    "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "DOHOL.IS",
    "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS",
    "HALKB.IS", "ISCTR.IS", "KCHOL.IS", "KRDMD.IS", "LOGO.IS",
    "MGROS.IS", "ODAS.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS",
    "SAHOL.IS", "SASA.IS", "SISE.IS", "TAVHL.IS", "TCELL.IS",
    "THYAO.IS", "TOASO.IS", "TTKOM.IS", "TUPRS.IS"
]

MARKET_TICKER = "XU100.IS"
STARTING_CAPITAL = 10000.0
TOP_K = 5
TRAIN_WINDOW = 252 * 2
MIN_HISTORY = 252

TRANSACTION_COST = 0.0015
SLIPPAGE_COST = 0.0005
TARGET_HORIZON = 1
HISTORY_FILE = "history.json"

# =============================================================================
# PERFORMANCE MATH EXTRACTION HELPERS
# =============================================================================
def safe_pct_change(series):
    return series.pct_change().replace([np.inf, -np.inf], np.nan)

def annualized_sharpe(returns):
    returns = pd.Series(returns).dropna()
    if len(returns) < 2 or returns.std() == 0: return np.nan
    return (returns.mean() / returns.std()) * np.sqrt(252)

def max_drawdown(equity_curve):
    equity = pd.Series(equity_curve)
    return ((equity - equity.cummax()) / equity.cummax()).min()

# =============================================================================
# LOAD VIRTUAL ACCOUNT ENVIRONMENT HISTORY
# =============================================================================
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f: history = json.load(f)
else:
    history = [{
        "date": "2026-05-19", "capital": STARTING_CAPITAL,
        "positions": ["CASH"], "weights": {"CASH": 1.0}, "daily_return": 0.0
    }]

last_entry = history[-1]
current_capital = float(last_entry["capital"])
last_positions = last_entry["positions"]
last_weights = last_entry.get("weights", {})
today_str = datetime.now().strftime("%Y-%m-%d")

# =============================================================================
# INGEST FINANCIAL MARKET CHANNELS
# =============================================================================
print("Downloading market data...")
ALL_TICKERS = BIST30_TICKERS + [MARKET_TICKER]
raw = yf.download(tickers=ALL_TICKERS, period="5y", auto_adjust=True, group_by="ticker", progress=False)

# --- Process Market Proxy ---
market_df = raw[MARKET_TICKER].copy()
market_close = market_df["Close"]
market_ret_1d = safe_pct_change(market_close)

market_features = pd.DataFrame(index=market_close.index)
market_features["mkt_ret_5d"] = market_close.pct_change(5)
market_features["mkt_ret_21d"] = market_close.pct_change(21)
market_features["mkt_vol_21d"] = market_ret_1d.rolling(21).std()
market_features["mkt_vol_63d"] = market_ret_1d.rolling(63).std()

# =============================================================================
# COMPUTE REALIZED PERFORMANCE FOR ACCOUNT CURVE LOGGING
# =============================================================================
daily_return = 0.0
if "CASH" not in last_positions and len(last_positions) > 0:
    weighted_return = 0.0
    valid_assets = 0
    for ticker in last_positions:
        if ticker not in raw.columns.levels[0]: continue
        df_t = raw[ticker].copy()
        if len(df_t) < 2: continue
        ret = safe_pct_change(df_t["Close"]).iloc[-1]
        if pd.isna(ret): continue
        weighted_return += ret * last_weights.get(ticker, 0)
        valid_assets += 1
    if valid_assets > 0:
        daily_return = weighted_return
        current_capital *= (1.0 + daily_return)

# =============================================================================
# EXTRACT CAUSAL PANEL INDICATORS (BUG FIX: SHIFT BEFORE ROLLING)
# =============================================================================
panel_data = []
live_rows = []

print("Engineering features...")
for ticker in BIST30_TICKERS:
    if ticker not in raw.columns.levels[0]: continue
    df = raw[ticker].copy()
    if len(df) < MIN_HISTORY: continue

    close, volume = df["Close"], df["Volume"]
    ret_1d = safe_pct_change(close)
    
    # Bug Fix: Enforce causal lookup barriers on indicators
    feats = pd.DataFrame(index=close.index)
    feats["ret_1d"] = ret_1d.shift(1)
    feats["ret_5d"] = close.pct_change(5).shift(1)
    feats["ret_10d"] = close.pct_change(10).shift(1)
    feats["ret_21d"] = close.pct_change(21).shift(1)
    feats["ret_63d"] = close.pct_change(63).shift(1)

    feats["vol_5d"] = ret_1d.shift(1).rolling(5).std()
    feats["vol_21d"] = ret_1d.shift(1).rolling(21).std()
    feats["vol_63d"] = ret_1d.shift(1).rolling(63).std()

    feats["intraday_range"] = ((df["High"] - df["Low"]) / close).shift(1)
    feats["volume_z"] = ((volume - volume.rolling(21).mean()) / volume.rolling(21).std()).shift(1)
    
    feats["skew_21d"] = ret_1d.shift(1).rolling(21).skew()
    feats["kurt_21d"] = ret_1d.shift(1).rolling(21).kurt()

    feats["trend_ma_ratio"] = (close.rolling(10).mean() / close.rolling(50).mean()).shift(1)
    feats["relative_strength_21d"] = (close.pct_change(21) - market_close.pct_change(21)).shift(1)

    # Combine data structures cleanly
    feats = feats.join(market_features.shift(1), how="left")
    
    # Continuous Regression target
    feats["target"] = close.pct_change(TARGET_HORIZON).shift(-TARGET_HORIZON)
    feats["ticker"] = ticker
    
    feats_clean = feats.dropna(subset=[c for c in feats.columns if c != "target"])
    
    # Store matrix blocks
    train_slice = feats_clean.dropna(subset=["target"])
    if not train_slice.empty:
        panel_data.append(train_slice)
        
    # Isolate final row representing today's closing metrics
    latest_row = feats_clean.iloc[[-1]].copy()
    live_rows.append(latest_row)

master = pd.concat(panel_data).sort_index()
feature_cols = [c for c in master.columns if c not in ["target", "ticker"]]
live_df = pd.concat(live_rows).set_index("ticker")

# Apply dynamic rolling training boundaries
recent_dates = sorted(master.index.unique())
if len(recent_dates) > TRAIN_WINDOW:
    master = master[master.index >= recent_dates[-TRAIN_WINDOW]]

# =============================================================================
# TRAIN MODEL PIPELINE
# =============================================================================
print("Training model...")
X_train, y_train = master[feature_cols], master["target"]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", HistGradientBoostingRegressor(max_depth=4, learning_rate=0.03, max_iter=250, min_samples_leaf=20, random_state=42))
])
model.fit(X_train, y_train)

# Generate Tomorrow's Forecast Metrics
live_df["expected_return"] = model.predict(live_df[feature_cols])

# =============================================================================
# EXECUTION LOGIC & INVERSE VOLATILITY VALUE TARGETING
# =============================================================================
selected = live_df.sort_values("expected_return", ascending=False).head(TOP_K)
selected = selected[selected["expected_return"] > 0]

if selected.empty:
    tomorrow_positions = ["CASH"]
    weights = {"CASH": 1.0}
else:
    # Scale inverse weights cleanly via variance
    vols = selected["vol_21d"].replace(0, np.nan)
    inv_vol = 1.0 / vols
    weights_raw = inv_vol / inv_vol.sum()
    
    weights = {ticker: float(w) for ticker, w in zip(selected.index, weights_raw)}
    tomorrow_positions = list(selected.index)

# =============================================================================
# ACCURATE PORTFOLIO LEVEL TURNOVER COST MODEL
# =============================================================================
# Bug Fix: Calculate transaction fee scale relative to actual traded position sizes
cost_rate_per_trade = TRANSACTION_COST + SLIPPAGE_COST
portfolio_cost_impact = 0.0

for asset in set(last_positions) | set(tomorrow_positions):
    old_w = last_weights.get(asset, 0.0)
    new_w = weights.get(asset, 0.0)
    # The true difference in capital allocation sizes
    portfolio_cost_impact += abs(new_w - old_w) * cost_rate_per_trade

current_capital *= (1.0 - portfolio_cost_impact)

# =============================================================================
# UPDATE TRACKER METRICS FOR THE LIVE INTERFACE
# =============================================================================
equity_curve = [x["capital"] for x in history] + [current_capital]
returns_series = [x.get("daily_return", 0) for x in history] + [daily_return]

sharpe = annualized_sharpe(returns_series)
mdd = max_drawdown(equity_curve)

new_entry = {
    "date": today_str, "capital": float(current_capital),
    "positions": tomorrow_positions, "weights": weights, "daily_return": float(daily_return),
    "sharpe": None if pd.isna(sharpe) else float(sharpe), "max_drawdown": float(mdd)
}

if history[-1]["date"] == today_str: history[-1] = new_entry
else: history.append(new_entry)

with open(HISTORY_FILE, "w") as f: json.dump(history, f, indent=2)

# =============================================================================
# OUTPUT DISPLAY PANELS
# =============================================================================
print("\n===================================================")
print("  STAT-ARB PERFORMANCE HUB SUMMARY ")
print("===================================================")
print(f"Date: {today_str} | Active Account Capital: {current_capital:,.2f} TRY")
print("\nTarget Portfolio Allocation for Tomorrow:")
if tomorrow_positions == ["CASH"]:
    print("   portfolios optimized to CASH liquidity reserves.")
else:
    for ticker in tomorrow_positions:
        print(f"   {ticker:<12} Predicted Return: {live_df.loc[ticker, 'expected_return']:>7.3%} | Target Weight: {weights[ticker]:>6.2%}")

print("\nRisk Parameters:")
print(f" Sharpe Ratio (Ann) : {sharpe:.2f}" if not pd.isna(sharpe) else " Sharpe Ratio: N/A")
print(f" Peak Max Drawdown  : {mdd:.2%}")
print("===================================================\n")
