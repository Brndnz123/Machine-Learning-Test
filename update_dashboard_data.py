"""
================================================================================
    update_dashboard_data.py (v21 - BIST 100 Macro & Volatility Fallback Release)
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
# ENGINE CONFIGURATION (UPGRADED TO COMPLETE BIST 100 UNIVERSE)
# =============================================================================
BIST100_TICKERS = [
    "AEFES.IS", "AGHOL.IS", "AKBNK.IS", "AKCNS.IS", "AKFGY.IS", "AKSA.IS", "AKSEN.IS", "ALARK.IS", "ALBRK.IS", "ALFAS.IS",
    "ARCLK.IS", "ASELS.IS", "ASTOR.IS", "ASUZU", "AYDEM.IS", "BAGFAS.IS", "BERA.IS", "BIMAS.IS", "BRSAN.IS", "BRYAT.IS",
    "BUCIM.IS", "CCOLA.IS", "CEMTS.IS", "CIMSA.IS", "CWENE.IS", "DOAS.IS", "DOHOL.IS", "ECILC.IS", "EGEEN.IS", "EKGYO.IS",
    "ENJSA.IS", "ENKAI.IS", "EREGL.IS", "EUPWR.IS", "FROTO.IS", "GARAN.IS", "GENIL.IS", "GESAN.IS", "GOLTS.IS", "GSDHO.IS",
    "GUBRF.IS", "GWIND.IS", "HALKB.IS", "HEKTS.IS", "IPEKE.IS", "ISCTR.IS", "ISGYO.IS", "ISMEN.IS", "IZMDC.IS", "KARDMD.IS",
    "KCAER.IS", "KCHOL.IS", "KENT.IS", "KONTR.IS", "KORDS.IS", "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "LOGO.IS", "MAVI.IS",
    "MGROS.IS", "MIATK.IS", "ODAS.IS", "OTKAR.IS", "OYAKC.IS", "PENTA.IS", "PETKM.IS", "PGSUS.IS", "QUAGR.IS", "SAHOL.IS",
    "SASA.IS", "SAYAS.IS", "SDTTR.IS", "SISE.IS", "SKBNK.IS", "SMRTG.IS", "SOKM.IS", "TABGD.IS", "TAVHL.IS", "TCELL.IS",
    "THYAO.IS", "TKFEN.IS", "TOASO.IS", "TSKB.IS", "TTKOM.IS", "TTRAK.IS", "TUKAS.IS", "TUPRS.IS", "TURSG.IS", "UFUK.IS",
    "ULKER.IS", "VAKBN.IS", "VESBE.IS", "VESTL.IS", "YEOTK.IS", "YKBNK.IS", "YYLGD.IS", "ZOREN.IS"
]

MARKET_TICKER = "XU100.IS"
FX_TICKER = "USDTRY=X"          

LOOKBACK_DAYS = 365 * 3         
TOP_K = 5
TARGET_HORIZON = 21             

STARTING_CAPITAL = 10000.0      
TRANSACTION_COST = 0.0015
SLIPPAGE_COST = 0.0005
MAX_SINGLE_POSITION_WEIGHT = 0.35
DRAWDOWN_CIRCUIT_BREAKER = -0.99  

ANNUAL_RISK_FREE_RATE = 0.45  
DAILY_RISK_FREE_RATE = (1.0 + ANNUAL_RISK_FREE_RATE) ** (1.0 / 252.0) - 1.0

# =============================================================================
# PIPELINE UTILITIES
# =============================================================================
def safe_pct_change(series):
    return series.pct_change().replace([np.inf, -np.inf], np.nan)

# =============================================================================
# DATA ACQUISITION & FEATURE GENERATION
# =============================================================================
print("Fetching real-time multi-asset BIST 100 matrix...")
end_dt = datetime.now()
start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)

ALL_TICKERS = BIST100_TICKERS + [MARKET_TICKER, FX_TICKER]
raw = yf.download(tickers=ALL_TICKERS, start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), auto_adjust=True, group_by="ticker", progress=False)

market_close = raw[MARKET_TICKER]["Close"].dropna()
market_ret_1d = safe_pct_change(market_close)
fx_close = raw[FX_TICKER]["Close"].dropna()

market_features = pd.DataFrame(index=market_close.index)
market_features["mkt_ret_5d"] = market_close.pct_change(5)
market_features["mkt_ret_21d"] = market_close.pct_change(21)
market_features["mkt_vol_21d"] = market_ret_1d.rolling(21).std()
market_features["mkt_vol_63d"] = market_ret_1d.rolling(63).std()
market_features["mkt_ma_10_50"] = market_close.rolling(10).mean() / market_close.rolling(50).mean()
market_features["mkt_ma_50_200"] = market_close.rolling(50).mean() / market_close.rolling(200).mean()
market_features["fx_ret_5d"] = fx_close.pct_change(5).reindex(market_features.index).ffill()
market_features["fx_ret_21d"] = fx_close.pct_change(21).reindex(market_features.index).ffill()

panel_list = []
asset_returns_matrix = pd.DataFrame()

print("Generating feature panel grids...")
for t in BIST100_TICKERS:
    if t not in raw.columns.levels[0] or raw[t].dropna(how='all').empty: continue
    df = raw[t].copy().dropna(subset=["Close"])
    
    close, volume = df["Close"], df["Volume"]
    ret_1d = safe_pct_change(close)
    asset_returns_matrix[t] = ret_1d
    
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
    feats["ticker"] = t
    
    panel_list.append(feats.dropna())

master_panel = pd.concat(panel_list).sort_index()
unique_dates = np.sort(master_panel.index.unique())

# =============================================================================
# INCREMENTAL LIVE TRAINING ENGINE LOOP
# =============================================================================
print("Calculating current alpha vectors across the BIST 100 index universe...")
train_dates = unique_dates[:-1]  
latest_date = unique_dates[-1]   

train_set = master_panel.loc[train_dates]
X_tr = train_set[[c for c in train_set.columns if c not in ["target", "ticker"]]]
y_tr = train_set["target"]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", HistGradientBoostingRegressor(max_depth=3, max_iter=100, learning_rate=0.03, random_state=42))
])
model.fit(X_tr.values, y_tr.values)

day_data = master_panel.loc[[latest_date]]
if day_data.empty:
    print(f"⚠️ Critical Error: Data signature for {latest_date} is empty. Aborting pipeline update.")
    exit(1)

X_step = day_data[[c for c in day_data.columns if c not in ["target", "ticker"]]]
tickers_step = day_data["ticker"].values
pred_alphas = model.predict(X_step.values)

rank_df = pd.DataFrame({
    "vol": day_data["vol_21d"].values,
    "rs": day_data["relative_strength_21d"].values
}, index=tickers_step)

alpha_std = pred_alphas.std()
rs_std = rank_df["rs"].std()

z_alpha = (pred_alphas - pred_alphas.mean()) / alpha_std if alpha_std > 0 else pred_alphas
z_rs = (rank_df["rs"] - rank_df["rs"].mean()) / rs_std if rs_std > 0 else rank_df["rs"]

rank_df["composite_score"] = z_alpha + z_rs
selected = rank_df.sort_values(by="composite_score", ascending=False).head(TOP_K)

# =============================================================================
# OPTIMIZED WEIGHT ALLOCATION WITH SAFE DATA FALLBACKS
# =============================================================================
vols = selected["vol"].replace(0, np.nan).dropna()

# Upgraded Fallback Guard: If less than 60% of the selection has clean volatility data, fall back to pure Equal Weight
if len(vols) < (TOP_K * 0.6):
    print("⚠️ Data anomaly caught: Insufficient volatility metrics. Deploying equal-weight configuration.")
    equal_weight = 1.0 / TOP_K
    next_positions = list(selected.index)
    next_weights = {t: float(equal_weight) for t in next_positions}
else:
    inv_vol = 1.0 / vols
    weights_raw = inv_vol / inv_vol.sum()
    
    # Align the computed inverse volatility series index back onto our top picks cleanly
    weights_aligned = pd.Series(0.0, index=selected.index)
    weights_aligned.update(weights_raw)
    
    clipped = np.minimum(weights_aligned, MAX_SINGLE_POSITION_WEIGHT)
    
    # Secure fallback check: if clipping forces a division by zero error, revert to equal weight instantly
    if clipped.sum() == 0:
        equal_weight = 1.0 / TOP_K
        next_positions = list(selected.index)
        next_weights = {t: float(equal_weight) for t in next_positions}
    else:
        renorm_weights = clipped / clipped.sum()
        next_positions = list(selected.index)
        next_weights = {t: float(w) for t, w in zip(selected.index, renorm_weights)}

# =============================================================================
# PARSE ACCOUNT EQUITY CURVE STATE & APPEND HISTORICAL LOGS
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
    last_positions_logged = ["CASH"]
    last_weights_logged = {"CASH": 1.0}
    running_peak = STARTING_CAPITAL
else:
    last_entry = history_data[-1]
    current_capital = last_entry["capital"]
    current_benchmark = last_entry.get("benchmark_capital", STARTING_CAPITAL)
    last_positions_logged = last_entry.get("positions", ["CASH"])
    last_weights_logged = last_entry.get("weights", {"CASH": 1.0})
    running_peak = max([e["capital"] for e in history_data] + [current_capital])

mkt_return_today = market_ret_1d.get(latest_date, 0.0)
if np.isnan(mkt_return_today): mkt_return_today = 0.0
current_benchmark *= (1.0 + mkt_return_today)

day_return = 0.0
if "CASH" not in last_positions_logged:
    for asset in last_positions_logged:
        try: asset_ret = asset_returns_matrix.loc[latest_date, asset]
        except KeyError: asset_ret = 0.0
        if np.isnan(asset_ret): asset_ret = 0.0
        day_return += asset_ret * last_weights_logged.get(asset, 0.0)
    current_capital *= (1.0 + day_return)

cost = 0.0
for asset in set(last_positions_logged) | set(next_positions):
    cost += abs(next_weights.get(asset, 0.0) - last_weights_logged.get(asset, 0.0)) * (TRANSACTION_COST + SLIPPAGE_COST)
current_capital *= (1.0 - cost)

current_drawdown = (current_capital - running_peak) / running_peak
if current_drawdown <= DRAWDOWN_CIRCUIT_BREAKER:
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

print(f"📊 Live Dashboard Pipeline execution completed successfully for: {new_record['date']}")
print(f"   Portfolio Value: {current_capital:,.2f} TRY | Benchmark Value: {current_benchmark:,.2f} TRY")
