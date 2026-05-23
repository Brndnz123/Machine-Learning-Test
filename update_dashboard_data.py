"""
================================================================================
    update_dashboard_data.py (v26 - Production v4 Ensemble Release)
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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# =============================================================================
# ENGINE CONFIGURATION (STABILIZED BIST UNIVERSE Matrix — 0% Download Errors)
# =============================================================================
BIST_SECTOR_MAP = {
    "AKBNK.IS": "Banks",    "GARAN.IS": "Banks",    "HALKB.IS": "Banks",
    "ISCTR.IS": "Banks",    "SKBNK.IS": "Banks",    "VAKBN.IS": "Banks",
    "YKBNK.IS": "Banks",    "ALBRK.IS": "Banks",    "TSKB.IS":  "Banks",
    "ISMEN.IS": "Banks",
    "SAHOL.IS": "Holdings", "KCHOL.IS": "Holdings", "AGHOL.IS": "Holdings",
    "DOHOL.IS": "Holdings", "GSDHO.IS": "Holdings", "SMRTG.IS": "Holdings",
    "AKSEN.IS": "Energy",   "AYDEM.IS": "Energy",   "ENJSA.IS": "Energy",
    "EUPWR.IS": "Energy",   "GWIND.IS": "Energy",   "CWENE.IS": "Energy",
    "ZOREN.IS": "Energy",   "ODAS.IS":  "Energy",   "TUPRS.IS": "Energy",
    "EREGL.IS": "Steel_Mining", "KRDMD.IS": "Steel_Mining", "BRSAN.IS": "Steel_Mining",
    "OYAKC.IS": "Steel_Mining",
    "AKSA.IS":  "Chemicals", "PETKM.IS": "Chemicals", "SASA.IS":  "Chemicals",
    "GUBRF.IS": "Chemicals",
    "CEMTS.IS": "Building_Mat", "BUCIM.IS": "Building_Mat",
    "AKCNS.IS": "Building_Mat", "CIMSA.IS": "Building_Mat",
    "SISE.IS":  "Building_Mat",
    "ENKAI.IS": "Real_Estate", "EKGYO.IS": "Real_Estate",
    "ISGYO.IS": "Real_Estate", "TKFEN.IS": "Real_Estate",
    "FROTO.IS": "Auto_Ind",  "TOASO.IS": "Auto_Ind",  "OTKAR.IS": "Auto_Ind",
    "TTRAK.IS": "Auto_Ind",  "ASUZU.IS": "Auto_Ind",  "ARCLK.IS": "Auto_Ind",
    "VESTL.IS": "Auto_Ind",  "VESBE.IS": "Auto_Ind",  "ASELS.IS": "Auto_Ind",
    "GESAN.IS": "Auto_Ind",  "GENIL.IS": "Auto_Ind",  "GOLTS.IS": "Auto_Ind",
    "KCAER.IS": "Auto_Ind",  "KONTR.IS": "Auto_Ind",  "KORDS.IS": "Auto_Ind",
    "ALFAS.IS": "Auto_Ind",  "HEKTS.IS": "Auto_Ind",
    "DOAS.IS":  "Consumer_Disc", "BRYAT.IS": "Consumer_Disc",
    "MAVI.IS":  "Consumer_Disc", "PENTA.IS": "Consumer_Disc",
    "BIMAS.IS": "Consumer_Stap", "MGROS.IS": "Consumer_Stap",
    "SOKM.IS":  "Consumer_Stap", "ULKER.IS": "Consumer_Stap",
    "CCOLA.IS": "Consumer_Stap", "TUKAS.IS": "Consumer_Stap",
    "AEFES.IS": "Consumer_Stap", "KENT.IS":  "Consumer_Stap",
    "YYLGD.IS": "Consumer_Stap", "TABGD.IS": "Consumer_Stap",
    "TCELL.IS": "Telecom_Tech", "TTKOM.IS": "Telecom_Tech",
    "LOGO.IS":  "Telecom_Tech", "SDTTR.IS": "Telecom_Tech",
    "ASTOR.IS": "Telecom_Tech", "MIATK.IS": "Telecom_Tech",
    "PGSUS.IS": "Transport", "TAVHL.IS": "Transport", "THYAO.IS": "Transport",
    "TURSG.IS": "Other", "QUAGR.IS": "Other", "SAYAS.IS": "Other",
    "UFUK.IS":  "Other", "EGEEN.IS": "Other", "ECILC.IS": "Other",
    "YEOTK.IS": "Other"
}

BIST100_TICKERS = list(BIST_SECTOR_MAP.keys())
MARKET_TICKER = "XU100.IS"
FX_TICKER = "USDTRY=X"          

LOOKBACK_DAYS = 365 * 3         
TOP_K = 5
TARGET_HORIZON = 21             

# ── v4 Advanced Risk Controls Bounds ─────────────────────────
VOL_TARGET = 0.20               # 20% annualized target volatility budget
MAX_GROSS_LEVERAGE = 1.25       # Leverage scaling multiplier cap
PARTICIPATION_LIMIT = 0.05      # Limited to 5% of daily ADV matrix
WINSORIZE_ALPHA = True

STARTING_CAPITAL = 100000.0     
TRANSACTION_COST = 0.0015
SLIPPAGE_COST = 0.0005
MAX_SINGLE_POSITION_WEIGHT = 0.35
DRAWDOWN_CIRCUIT_BREAKER = -0.99  

MIN_ADV_TRY = 5000000.0
MIN_ACTIVE_DAY_RATIO = 0.80
VOL_FLOOR = 0.01

ANNUAL_RISK_FREE_RATE = 0.45  
DAILY_RISK_FREE_RATE = (1.0 + ANNUAL_RISK_FREE_RATE) ** (1.0 / 252.0) - 1.0

# =============================================================================
# DATA PIPELINE ACQUISITION DESK
# =============================================================================
print("Fetching operational multi-asset data matrices from yFinance...")
end_dt = datetime.now()
start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)

ALL_TICKERS = BIST100_TICKERS + [MARKET_TICKER, FX_TICKER]
raw = yf.download(tickers=ALL_TICKERS, start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), auto_adjust=True, group_by="ticker", progress=False)

market_close = raw[MARKET_TICKER]["Close"].dropna()
market_ret_1d = market_close.pct_change().replace([np.inf, -np.inf], np.nan)
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

print("Constructing causal parameter factors...")
for t in BIST100_TICKERS:
    try: df = raw[t].copy().dropna(subset=["Close"])
    except KeyError: continue
    if df.empty: continue
    
    close, volume = df["Close"], df["Volume"]
    ret_1d = close.pct_change().replace([np.inf, -np.inf], np.nan)
    asset_returns_matrix[t] = ret_1d
    
    feats = pd.DataFrame(index=close.index)
    for lag in [1, 5, 10, 21, 63]:
        feats[f"ret_{lag}d"] = close.pct_change(lag).shift(1)
    for lag in [5, 21, 63]:
        feats[f"vol_{lag}d"] = ret_1d.rolling(lag).std().shift(1)
        
    feats["skew_21d"] = ret_1d.rolling(21).skew().shift(1)
    feats["kurt_21d"] = ret_1d.rolling(21).kurt().shift(1)
    feats["intraday_range"] = ((df["High"] - df["Low"]) / close).shift(1)
    
    vol_std = volume.rolling(21).std().replace(0, np.nan)
    feats["volume_z"] = ((volume - volume.rolling(21).mean()) / vol_std).shift(1)
    feats["trend_ma_ratio"] = (close.rolling(10).mean() / close.rolling(50).mean()).shift(1)
    
    # Secure alignment prevents index shifts from throwing NaN cascades
    aligned_mkt = market_close.pct_change(21).reindex(close.index).ffill()
    feats["relative_strength_21d"] = (close.pct_change(21) - aligned_mkt).shift(1)
    feats["adv_63d_try"] = (close * volume).rolling(63).mean().shift(1)
    
    feats = feats.join(market_features.shift(1), how="left")
    feats["target"] = close.pct_change(TARGET_HORIZON).shift(-TARGET_HORIZON)
    feats["ticker"] = t
    
    panel_list.append(feats)

master_panel = pd.concat(panel_list).sort_index()
unique_dates = np.sort(master_panel.index.unique())

# =============================================================================
# ENSEMBLE SYSTEM INFERENCE DESK
# =============================================================================
print("Running production walk-forward ensemble alignment...")
train_dates = unique_dates[:-1]
latest_date = unique_dates[-1]

train_set = master_panel.loc[train_dates].dropna(subset=["target"])
feature_cols = [c for c in master_panel.columns if c not in {"target", "ticker", "adv_63d_try"}]

# v4 Blend Architecture Matrix: 60% HistGBM + 40% Random Forest
ensemble = VotingRegressor(estimators=[
    ("gbm", HistGradientBoostingRegressor(max_depth=3, max_iter=100, learning_rate=0.03, random_state=42)),
    ("rf", RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1))
], weights=[0.6, 0.4])

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("ensemble", ensemble)
])
model.fit(train_set[feature_cols].values, train_set["target"].values)

day_panel = master_panel.loc[[latest_date]].copy()

# Enforce liquidity filters
day_panel = day_panel[day_panel["adv_63d_try"] >= MIN_ADV_TRY]
train_window_returns = asset_returns_matrix.loc[asset_returns_matrix.index.isin(train_dates)]
active_ratio = train_window_returns.notna().mean()
liquid_tickers = active_ratio[active_ratio >= MIN_ACTIVE_DAY_RATIO].index
day_panel = day_panel[day_panel["ticker"].isin(liquid_tickers)]

if day_panel.empty:
    next_positions = ["CASH"]
    next_weights = {"CASH": 1.0}
else:
    fcols = [c for c in day_panel.columns if c not in {"target", "ticker", "adv_63d_try"}]
    tickers_now = day_panel["ticker"].values
    
    X_live = day_panel[fcols].reindex(columns=fcols)
    preds = model.predict(X_live.values)
    
    alpha_series = pd.Series(preds, index=tickers_now)
    if WINSORIZE_ALPHA:
        lo, hi = alpha_series.quantile(0.02), alpha_series.quantile(0.98)
        alpha_series = alpha_series.clip(lo, hi)
        
    rank_df = pd.DataFrame({
        "vol_21d": day_panel["vol_21d"].values,
        "rs": day_panel["relative_strength_21d"].values
    }, index=tickers_now)
    
    def get_zscore(s):
        return (s - s.mean()) / s.std() if s.std() > 0 else s

    z_rs = get_zscore(pd.Series(rank_df["rs"].values, index=tickers_now))
    z_alpha = get_zscore(alpha_series)
    
    rank_df["score"] = z_alpha + z_rs
    selected = rank_df.nlargest(TOP_K, "score")
    
    vols = selected["vol_21d"].replace(0, np.nan).dropna()
    if len(vols) < max(1, int(TOP_K * 0.6)):
        next_positions = list(selected.index)
        next_weights = {t: 1.0 / TOP_K for t in next_positions}
    else:
        vols_floored = vols.clip(lower=VOL_FLOOR)
        inv_vol = 1.0 / vols_floored
        raw_weights = inv_vol / inv_vol.sum()
        
        # ── v4 Volatility targeting scale sizer ───────────
        port_ann_vol = float(vols_floored.mean() * np.sqrt(252))
        leverage_scale = VOL_TARGET / port_ann_vol if port_ann_vol > 0 else 1.0
        leverage_scale = min(leverage_scale, MAX_GROSS_LEVERAGE)
        
        scaled_weights = raw_weights * leverage_scale
        clipped_weights = scaled_weights.clip(upper=MAX_SINGLE_POSITION_WEIGHT)
        
        if clipped_weights.sum() == 0:
            next_positions = list(selected.index)
            next_weights = {t: 1.0 / TOP_K for t in next_positions}
        else:
            next_weights = {t: float(w) for t, w in clipped_weights.items()}
            
            # ── v4 Liquidity Participation Limits Cap ───────────
            for asset in list(next_weights.keys()):
                try:
                    adv_val = float(day_panel[day_panel["ticker"] == asset]["adv_63d_try"].values[0])
                    max_trade_size = adv_val * PARTICIPATION_LIMIT
                    allocated_funds = next_weights[asset] * STARTING_CAPITAL 
                    if allocated_funds > max_trade_size:
                        next_weights[asset] = max_trade_size / STARTING_CAPITAL
                except:
                    pass
            next_positions = list(next_weights.keys())

# =============================================================================
# ACCOUNT BALANCE HISTORY ENGINE
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
        
    cash_remainder = 1.0 - sum(last_weights.values())
    if cash_remainder > 0:
        day_return += DAILY_RISK_FREE_RATE * cash_remainder
        
    current_capital *= (1.0 + day_return)

# Turnover cost friction deductions
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
    "max_drawdown": float(min([e.get("max_drawdown", 0.0) for e in history_data] + [current_dd])),
    "positions": next_positions,
    "weights": next_weights
}

history_data.append(new_record)
if len(history_data) > 250:
    history_data = history_data[-250:]

with open(history_file, "w") as f:
    json.dump(history_data, f, indent=4)

print(f"📊 Production v4 Engine synchronized successfully for: {new_record['date']}")
print(f"   Target Targets: {next_positions}")
print(f"   Portfolio Capital: {current_capital:,.2f} TRY | Index Benchmark: {current_benchmark:,.2f} TRY")
