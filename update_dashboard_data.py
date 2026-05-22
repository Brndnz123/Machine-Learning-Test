"""
================================================================================
    update_dashboard_data.py (v23 - Production T+1 Execution Release)
    Automated execution script for GitHub Actions deployment pipeline.
================================================================================
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# =============================================================================
# ENGINE CONFIGURATION (FULLY ALIGNED TO V2 EXECUTION LAB)
# =============================================================================
BIST100_TICKERS = [
    "AKBNK.IS", "ASELS.IS", "BIMAS.IS", "EREGL.IS",
    "FROTO.IS", "GARAN.IS", "ISCTR.IS", "KCHOL.IS",
    "MGROS.IS", "PGSUS.IS", "SAHOL.IS",
    "SISE.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS",
    "TUPRS.IS", "YKBNK.IS"
]

MARKET_TICKER = "XU100.IS"
FX_TICKER = "USDTRY=X"          

LOOKBACK_DAYS = 365 * 3         
TOP_K = 5
TARGET_HORIZON = 21             

STARTING_CAPITAL = 100000.0     # Reset to 100k to match your realistic research base
TRANSACTION_COST = 0.0015
SLIPPAGE_COST = 0.0005
MAX_SINGLE_POSITION_WEIGHT = 0.30
MAX_SECTOR_WEIGHT = 0.40
MIN_ADV_TRY = 5000000.0
VOLATILITY_FLOOR = 0.01
DRAWDOWN_CIRCUIT_BREAKER = -0.35  

ANNUAL_RISK_FREE_RATE = 0.45  
DAILY_RISK_FREE_RATE = (1.0 + ANNUAL_RISK_FREE_RATE) ** (1.0 / 252.0) - 1.0

SECTOR_MAP = {
    "AKBNK.IS": "BANK", "GARAN.IS": "BANK", "ISCTR.IS": "BANK", "YKBNK.IS": "BANK",
    "KCHOL.IS": "HOLDING", "SAHOL.IS": "HOLDING",
    "THYAO.IS": "TRANSPORT", "PGSUS.IS": "TRANSPORT",
    "TUPRS.IS": "ENERGY",
    "BIMAS.IS": "RETAIL", "MGROS.IS": "RETAIL",
    "ASELS.IS": "DEFENSE",
    "EREGL.IS": "STEEL",
    "FROTO.IS": "AUTO", "TOASO.IS": "AUTO",
    "TCELL.IS": "TELCO",
    "SISE.IS": "INDUSTRIAL"
}

# =============================================================================
# DATA PIPELINE ACQUISITION
# =============================================================================
print("Fetching live operational matrices from Yahoo Finance...")
end_dt = datetime.now()
start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)

ALL_TICKERS = BIST100_TICKERS + [MARKET_TICKER, FX_TICKER]
raw = yf.download(tickers=ALL_TICKERS, start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), auto_adjust=True, group_by="ticker", progress=False)

market_close = raw[MARKET_TICKER]["Close"].ffill()
market_ret_1d = market_close.pct_change()

panel_list = []
asset_returns_matrix = pd.DataFrame()

print("Processing causal indicator tables...")
for t in BIST100_TICKERS:
    if t not in raw.columns.levels[0] or raw[t].dropna(how='all').empty: continue
    df = raw[t].copy().dropna(subset=["Close"])
    
    close, volume = df["Close"], df["Volume"]
    ret_1d = close.pct_change()
    asset_returns_matrix[t] = ret_1d
    
    feats = pd.DataFrame(index=close.index)
    feats["ret_5d"] = close.pct_change(5).shift(1)
    feats["ret_21d"] = close.pct_change(21).shift(1)
    feats["ret_63d"] = close.pct_change(63).shift(1)
    feats["vol_21d"] = ret1d.rolling(21).std().shift(1)
    feats["ma_ratio"] = (close.rolling(10).mean() / close.rolling(50).mean()).shift(1)
    
    aligned_mkt_ret = market_close.pct_change(21).reindex(close.index).ffill()
    feats["relative_strength"] = (close.pct_change(21) - aligned_mkt_ret).shift(1)
    feats["adv_try"] = (close * volume).rolling(21).mean().shift(1)
    
    feats["target"] = close.pct_change(TARGET_HORIZON).shift(-TARGET_HORIZON)
    feats["ticker"] = t
    feats["sector"] = SECTOR_MAP.get(t, "OTHER")
    
    panel_list.append(feats.dropna())

master_panel = pd.concat(panel_list).sort_index()
unique_dates = np.sort(master_panel.index.unique())

# =============================================================================
# ROLLING WALK-FORWARD INFERENCE
# =============================================================================
print("Calculating upcoming portfolio deployment targets...")
train_dates = unique_dates[:-1]
latest_date = unique_dates[-1]

train_set = master_panel.loc[train_dates]
feature_cols = [c for c in master_panel.columns if c not in {"target", "ticker", "sector"}]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", HistGradientBoostingRegressor(max_depth=3, max_iter=100, learning_rate=0.03, random_state=42))
])
model.fit(train_set[feature_cols].values, train_set["target"].values)

today_data = master_panel.loc[[latest_date]].copy()
X_live = today_data[feature_cols].reindex(columns=feature_cols)
today_data["alpha"] = model.predict(X_live.values)

def get_zscore(x):
    std = x.std()
    return (x - x.mean()) / std if std > 0 else pd.Series(0, index=x.index)

today_data["score"] = get_zscore(today_data["alpha"]) + get_zscore(today_data["relative_strength"])
today_data = today_data.set_index("ticker")

# Apply Liquidity ADV Screening Filters
today_data = today_data[today_data["adv_try"] > MIN_ADV_TRY]

if today_data.empty:
    next_positions = ["CASH"]
    next_weights = {"CASH": 1.0}
else:
    selected = today_data.nlargest(TOP_K, "score").copy()
    selected["vol_21d"] = selected["vol_21d"].clip(lower=VOLATILITY_FLOOR)
    inv_vol = 1.0 / selected["vol_21d"]
    
    raw_weights = inv_vol / inv_vol.sum()
    selected["weight"] = raw_weights
    selected["weight"] = np.minimum(selected["weight"], MAX_SINGLE_POSITION_WEIGHT)
    
    for sector in selected["sector"].unique():
        mask = selected["sector"] == sector
        sector_sum = selected.loc[mask, "weight"].sum()
        if sector_sum > MAX_SECTOR_WEIGHT:
            selected.loc[mask, "weight"] *= (MAX_SECTOR_WEIGHT / sector_sum)
            
    selected["weight"] /= selected["weight"].sum()
    next_positions = list(selected.index)
    next_weights = {t: float(w) for t, w in zip(selected.index, selected["weight"])}

# =============================================================================
# ACCOUNT ACCOUNTING & DATABASE MUTATION DESK
# =============================================================================
history_file = "history.json"
if os.path.exists(history_file):
    with open(history_file, "r") as f:
        try: history_data = json.load(f)
        except json.JSONDecodeError: history_data = []
else:
    history_data = []

if not history_data:
    current_capital = STARTING_CAPITAL
    current_benchmark = STARTING_CAPITAL
    last_positions = ["CASH"]
    last_weights = {"CASH": 1.0}
    running_peak = STARTING_CAPITAL
else:
    last_entry = history_data[-1]
    current_capital = last_entry["capital"]
    current_benchmark = last_entry.get("benchmark_capital", STARTING_CAPITAL)
    last_positions = last_entry.get("positions", ["CASH"])
    last_weights = last_entry.get("weights", {"CASH": 1.0})
    running_peak = max([e["capital"] for e in history_data] + [current_capital])

# Accrue asset returns using current weights
mkt_return = market_ret_1d.get(latest_date, 0.0)
if np.isnan(mkt_return): mkt_return = 0.0
current_benchmark *= (1.0 + mkt_return)

day_return = 0.0
if "CASH" not in last_positions:
    for asset in last_positions:
        try: r = asset_returns_matrix.loc[latest_date, asset]
        except KeyError: r = 0.0
        if np.isnan(r): r = 0.0
        day_return += r * last_weights.get(asset, 0.0)
    current_capital *= (1.0 + day_return)

# Deduct Turnover Cost Frictions
all_assets = set(last_weights) | set(next_weights)
turnover = sum(abs(next_weights.get(a, 0.0) - last_weights.get(a, 0.0)) for a in all_assets)
current_capital *= (1.0 - turnover * (TRANSACTION_COST + SLIPPAGE_COST))

current_dd = (current_capital - running_peak) / running_peak
if current_dd <= DRAWDOWN_CIRCUIT_BREAKER:
    next_positions = ["CASH"]
    next_weights = {"CASH": 1.0}

if len(history_data) > 5:
    cap_series = pd.Series([e["capital"] for e in history_data] + [current_capital])
    ret_series = cap_series.pct_change().dropna()
    excess_ret = ret_series - DAILY_RISK_FREE_RATE
    excess_std = excess_ret.std()
    computed_sharpe = (excess_ret.mean() / excess_std) * np.sqrt(252) if excess_std > 0 else 0.0
else:
    computed_sharpe = 0.0

new_record = {
    "date": pd.Timestamp(latest_date).strftime("%Y-%m-%d"),
    "capital": float(current_capital),
    "benchmark_capital": float(current_benchmark),
    "sharpe": float(computed_sharpe),
    "max_drawdown": float(min([e.get("max_drawdown", 0.0) for e in history_data] + [current_drawdown])),
    "positions": next_positions,
    "weights": next_weights
}

history_data.append(new_record)
if len(history_data) > 250:
    history_data = history_data[-250:]

with open(history_file, "w") as f:
    json.dump(history_data, f, indent=4)

print(f"📊 Production state machine synchronized successfully for date: {new_record['date']}")
print(f"   Portfolio Capital Base: {current_capital:,.2f} TRY | Benchmark Index Base: {current_benchmark:,.2f} TRY")
