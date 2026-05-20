"""
================================================================================
    update_dashboard_data.py (v8 - Audited Production Release)
================================================================================
"""
import os
import json
import warnings
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# CONFIGURATION
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

# Institutional Risk Settings
ANNUAL_RISK_FREE_RATE = 0.45  
DAILY_RISK_FREE_RATE = (1.0 + ANNUAL_RISK_FREE_RATE) ** (1.0 / 252.0) - 1.0
MAX_SINGLE_POSITION_WEIGHT = 0.35  
DRAWDOWN_CIRCUIT_BREAKER = -0.15   

# =============================================================================
# MATH CORE & RISK HELPERS
# =============================================================================
def safe_pct_change(series):
    return series.pct_change().replace([np.inf, -np.inf], np.nan)

def annualized_sharpe(returns_list, daily_rf):
    # Filter out cold-start 0.0 metrics to prevent inflation of the ratio
    clean_returns = [r for r in returns_list if r != 0.0]
    ret_series = pd.Series(clean_returns).dropna()
    if len(ret_series) < 2: return np.nan
    
    excess_returns = ret_series - daily_rf
    excess_std = excess_returns.std()
    
    if excess_std == 0: return np.nan
    return (excess_returns.mean() / excess_std) * np.sqrt(252)

def calculate_drawdown_state(equity_history, current_val):
    full_curve = pd.Series(equity_history + [current_val])
    return ((full_curve - full_curve.cummax()) / full_curve.cummax()).iloc[-1]

# =============================================================================
# STATE SYNCHRONIZATION
# =============================================================================
today_str = datetime.now().strftime("%Y-%m-%d")

if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f: history = json.load(f)
else:
    history = [{
        "date": today_str, "capital": STARTING_CAPITAL, "benchmark_capital": STARTING_CAPITAL,
        "positions": ["CASH"], "weights": {"CASH": 1.0}, "daily_return": 0.0, "benchmark_return": 0.0
    }]

last_entry = history[-1]
current_capital = float(last_entry["capital"])
current_benchmark = float(last_entry.get("benchmark_capital", STARTING_CAPITAL))
last_positions = last_entry["positions"]
last_weights = last_entry.get("weights", {})
past_equity_curve = [x["capital"] for x in history[:-1]] if len(history) > 1 else [STARTING_CAPITAL]

# =============================================================================
# INGESTION & DATA CLEANING
# =============================================================================
logging.info("Downloading multi-asset historical matrix...")
ALL_TICKERS = BIST30_TICKERS + [MARKET_TICKER]
raw = yf.download(tickers=ALL_TICKERS, period="5y", auto_adjust=True, group_by="ticker", progress=False)

available_tickers = []
for t in BIST30_TICKERS:
    if t in raw.columns.levels[0] and not raw[t].dropna(how='all').empty:
        available_tickers.append(t)
    else:
        logging.warning(f"Ticker Asset {t} unavailable or dropped.")

# =============================================================================
# SYNCED CIRCUIT BREAKER MATRIX CALCULATIONS
# =============================================================================
# Calculate the exact current drawdown before checking performance updates
circuit_breaker_triggered = False
pre_update_mdd = calculate_drawdown_state(past_equity_curve, current_capital)

if pre_update_mdd <= DRAWDOWN_CIRCUIT_BREAKER:
    logging.critical(f"🚨 CIRCUIT BREAKER BREACHED ({pre_update_mdd:.2%}). HALTING OPERATIONS.")
    circuit_breaker_triggered = True

# --- Update Index Benchmark ---
market_df = raw[MARKET_TICKER].copy().dropna(subset=["Close"])
mkt_return = 0.0
if len(market_df) >= 2:
    mkt_return = safe_pct_change(market_df["Close"]).iloc[-1]
    if np.isnan(mkt_return): mkt_return = 0.0
current_benchmark *= (1.0 + mkt_return)

# --- Accrue Realized Returns (Only if circuit breaker is clear) ---
daily_return = 0.0
if "CASH" not in last_positions and len(last_positions) > 0 and not circuit_breaker_triggered:
    weighted_return = 0.0
    valid_assets = 0
    for ticker in last_positions:
        if ticker not in available_tickers: continue
        df_t = raw[ticker].copy()
        if len(df_t) < 2: continue
        ret = safe_pct_change(df_t["Close"]).iloc[-1]
        if pd.isna(ret): continue
        weighted_return += ret * last_weights.get(ticker, 0.0)
        valid_assets += 1
    if valid_assets > 0:
        daily_return = weighted_return
        current_capital *= (1.0 + daily_return)

# =============================================================================
# CAUSAL FEATURE PIPELINE GENERATOR
# =============================================================================
market_close = market_df["Close"]
market_ret_1d = safe_pct_change(market_close)

market_features = pd.DataFrame(index=market_close.index)
market_features["mkt_ret_5d"] = market_close.pct_change(5)
market_features["mkt_ret_21d"] = market_close.pct_change(21)
market_features["mkt_vol_21d"] = market_ret_1d.rolling(21).std()
market_features["mkt_vol_63d"] = market_ret_1d.rolling(63).std()

panel_data = []
live_rows_dict = {}

for ticker in available_tickers:
    df = raw[ticker].copy()
    if len(df) < MIN_HISTORY: continue

    close, volume = df["Close"], df["Volume"]
    ret_1d = safe_pct_change(close)
    
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

    feats = feats.join(market_features.shift(1), how="left")
    feats["target"] = close.pct_change(TARGET_HORIZON).shift(-TARGET_HORIZON)
    feats["ticker"] = ticker
    
    feats_clean = feats.dropna(subset=[c for c in feats.columns if c != "target"])
    train_slice = feats_clean.dropna(subset=["target"])
    
    if not train_slice.empty: 
        panel_data.append(train_slice)
        
    # Guard against duplicate ticker allocation errors by caching via dictionary overwrite
    live_rows_dict[ticker] = feats_clean.iloc[-1].to_dict()

master = pd.concat(panel_data).sort_index()
feature_cols = [c for c in master.columns if c not in ["target", "ticker"]]

# Convert clean dictionary mapping to data frame seamlessly
live_df = pd.DataFrame.from_dict(live_rows_dict, orient='index')

# Restrict the timeline memory window
recent_dates = sorted(master.index.unique())
if len(recent_dates) > TRAIN_WINDOW:
    master = master[master.index >= recent_dates[-TRAIN_WINDOW]]

# =============================================================================
# TRAIN MODEL & ALLOCATION DISCIPLINE
# =============================================================================
X_train, y_train = master[feature_cols], master["target"]
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", HistGradientBoostingRegressor(max_depth=4, learning_rate=0.03, max_iter=250, min_samples_leaf=20, random_state=42))
])
model.fit(X_train, y_train)
live_df["expected_return"] = model.predict(live_df[feature_cols])

# =============================================================================
# ENFORCE FINAL ALROCATION AND TRANSACTION IMPACTS
# =============================================================================
final_equity_curve = [x["capital"] for x in history] + [current_capital]
post_update_mdd = max_drawdown(final_equity_curve)

if circuit_breaker_triggered or post_update_mdd <= DRAWDOWN_CIRCUIT_BREAKER:
    tomorrow_positions = ["CASH"]
    weights = {"CASH": 1.0}
else:
    selected = live_df.sort_values("expected_return", ascending=False).head(TOP_K)
    selected = selected[selected["expected_return"] > 0]

    if selected.empty:
        tomorrow_positions = ["CASH"]
        weights = {"CASH": 1.0}
    else:
        vols = selected["vol_21d"].replace(0, np.nan)
        inv_vol = 1.0 / vols
        weights_raw = inv_vol / inv_vol.sum()
        
        # Enforce the concentration cap with post-clipping renormalization
        clipped_weights = np.minimum(weights_raw, MAX_SINGLE_POSITION_WEIGHT)
        renormalized_weights = clipped_weights / clipped_weights.sum()
        
        weights = {ticker: float(w) for ticker, w in zip(selected.index, renormalized_weights)}
        tomorrow_positions = list(selected.index)

# Friction Accounting Engine
portfolio_cost_impact = 0.0
for asset in set(last_positions) | set(tomorrow_positions):
    old_w = last_weights.get(asset, 0.0)
    new_w = weights.get(asset, 0.0)
    portfolio_cost_impact += abs(new_w - old_w) * (TRANSACTION_COST + SLIPPAGE_COST)

current_capital *= (1.0 - portfolio_cost_impact)

# =============================================================================
# PERFORMANCE TRACKING EXPORTS
# =============================================================================
all_historical_returns = [x.get("daily_return", 0.0) for x in history] + [daily_return]
sharpe = annualized_sharpe(all_historical_returns, DAILY_RISK_FREE_RATE)

new_entry = {
    "date": today_str, "capital": float(current_capital), "benchmark_capital": float(current_benchmark),
    "positions": tomorrow_positions, "weights": weights, "daily_return": float(daily_return),
    "benchmark_return": float(mkt_return), "sharpe": None if pd.isna(sharpe) else float(sharpe), "max_drawdown": float(post_update_mdd)
}

if history[-1]["date"] == today_str: history[-1] = new_entry
else: history.append(new_entry)

with open(HISTORY_FILE, "w") as f: json.dump(history, f, indent=2)
logging.info(f"Process complete. Allocation targets for tomorrow session: {tomorrow_positions}")
