"""
================================================================================
  update_dashboard_data.py
  This is the live production script executed daily inside GitHub Actions.
================================================================================
"""
import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy.fft import fft, fftfreq
from sklearn.ensemble import HistGradientBoostingClassifier

# --- Config ---
BIST30_TICKERS = [
    "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "DOHOL.IS",
    "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS",
    "HALKB.IS", "ISCTR.IS", "KCHOL.IS", "KRDMD.IS", "LOGO.IS",
    "MGROS.IS", "ODAS.IS",  "OYAKC.IS", "PETKM.IS", "PGSUS.IS",
    "SAHOL.IS", "SASA.IS",  "SISE.IS",  "TAVHL.IS", "TCELL.IS",
    "THYAO.IS", "TOASO.IS", "TTKOM.IS", "TUPRS.IS"
]

FFT_WINDOW = 252   
N_TOP_FREQS = 3     
VOL_WINDOW = 21    
LAG_DAYS = [1, 2, 3, 5, 21]
SIGMA_THRESHOLD = 0.5
TOP_K_ASSETS = 3
FEES_PER_SIDE = 0.0005

# ── 1. LOAD OR INITIALIZE VIRTUAL ACC COUNT HISTORY ──────────────────────────
HISTORY_FILE = "history.json"
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
else:
    # First time initialization anchor
    history = [{"date": "2026-05-19", "capital": 10000.00, "positions": ["CASH"]}]

last_entry = history[-1]
current_capital = last_entry["capital"]
last_held_positions = last_entry["positions"]

# ── 2. DOWNLOAD MAXIMUM RECENT DATA EXTENTS ───────────────────────────────────
raw = yf.download(tickers=BIST30_TICKERS, period="3y", auto_adjust=False, group_by="ticker", progress=False)

# ── 3. COMPUTE PERFORMANCE OF YESTERDAY'S ALLOCATION ──────────────────────────
today_str = datetime.now().strftime("%Y-%m-%d")

if "CASH" not in last_held_positions and len(last_held_positions) > 0:
    returns_sum = 0.0
    actual_traded_count = 0
    
    for ticker in last_held_positions:
        if ticker in raw.columns.levels[0]:
            df_t = raw[ticker]
            # Calculate today's closing performance percentage shift
            if len(df_t) >= 2:
                pct_change = df_t["Adj Close"].pct_change().iloc[-1]
                if not np.isnan(pct_change):
                    returns_sum += pct_change
                    actual_traded_count += 1
                    
    if actual_traded_count > 0:
        mean_return = returns_sum / actual_traded_count
        current_capital *= (1.0 + mean_return)

# ── 4. RUN PIPELINE MODEL FORECASTER FOR TOMORROW ─────────────────────────────
panel_frames = []
live_features_today = []

for ticker in BIST30_TICKERS:
    if ticker not in raw.columns.levels[0]: continue
    df_t = raw[ticker].copy().dropna(subset=["Adj Close"])
    cb_mask = ((df_t["High"] == df_t["Low"]) | (df_t["Volume"] == 0))
    df_t = df_t[~cb_mask].copy()
    
    log_ret = np.log(df_t["Adj Close"] / df_t["Adj Close"].shift(1))
    rolling_vol = log_ret.shift(1).rolling(VOL_WINDOW, min_periods=10).std().replace(0, np.nan)
    z_ret = (log_ret / rolling_vol).dropna()
    
    if len(z_ret) < FFT_WINDOW + 2: continue
    arr, dates = z_ret.values, z_ret.index
    
    # Structural training records extraction
    fft_rows = []
    for i in range(FFT_WINDOW, len(arr)):
        w = arr[i - FFT_WINDOW : i]
        vals = fft(w)
        amps, freqs = np.abs(vals[: len(w) // 2]), fftfreq(len(w), d=1.0)[: len(w) // 2]
        amps[0] = 0.0
        top = np.argsort(amps)[::-1][:N_TOP_FREQS]
        row = {}
        for r, idx in enumerate(top, start=1):
            row[f"fft_amp_{r}"] = amps[idx]
            row[f"fft_freq_{r}"] = freqs[idx]
            row[f"fft_period_{r}"] = (1.0 / freqs[idx]) if freqs[idx] > 1e-9 else np.nan
        row["date"] = dates[i]
        fft_rows.append(row)
        
    fft_df = pd.DataFrame(fft_rows).set_index("date")
    lag_df = pd.DataFrame(index=z_ret.index)
    for k in LAG_DAYS: lag_df[f"z_lag_{k}d"] = z_ret.shift(k)
    lag_df["z_rvol_21d"] = z_ret.shift(1).rolling(21).std()
    lag_df["z_rskew_21d"] = z_ret.shift(1).rolling(21).skew()
    
    feats = fft_df.join(lag_df, how="left").dropna()
    next_z = z_ret.shift(-1).reindex(feats.index)
    target = pd.Series(np.nan, index=feats.index)
    target[next_z > SIGMA_THRESHOLD] = 1
    target[next_z < -SIGMA_THRESHOLD] = 0
    
    historical_idx = feats.index.intersection(target.dropna().index)
    hist_feats = feats.loc[historical_idx]
    hist_feats["target"] = target.loc[historical_idx].values
    hist_feats["ticker"] = ticker
    panel_frames.append(hist_feats)
    
    # Today's live record feature alignment block
    w_live = arr[-FFT_WINDOW:]
    v_live = fft(w_live)
    a_live, f_live = np.abs(v_live[: len(w_live) // 2]), fftfreq(len(w_live), d=1.0)[: len(w_live) // 2]
    a_live[0] = 0.0
    top_l = np.argsort(a_live)[::-1][:N_TOP_FREQS]
    
    live_row = {}
    for r, idx in enumerate(top_l, start=1):
        live_row[f"fft_amp_{r}"] = a_live[idx]
        live_row[f"fft_freq_{r}"] = f_live[idx]
        live_row[f"fft_period_{r}"] = (1.0 / f_live[idx]) if f_live[idx] > 1e-9 else np.nan
    for k in LAG_DAYS: live_row[f"z_lag_{k}d"] = z_ret.iloc[-k]
    live_row["z_rvol_21d"] = z_ret.iloc[-21:].std()
    live_row["z_rskew_21d"] = z_ret.iloc[-21:].skew()
    live_row["ticker"] = ticker
    live_features_today.append(live_row)

master_panel = pd.concat(panel_frames, axis=0)
live_df = pd.DataFrame(live_features_today).set_index("ticker")

for ticker in BIST30_TICKERS:
    col_name = f"is_{ticker}"
    master_panel[col_name] = (master_panel["ticker"] == ticker).astype(float)
    live_df[col_name] = (live_df.index == ticker).astype(float)

exclude_cols = {"ticker", "target", "raw_next_return"}
feat_cols = [c for c in master_panel.columns if c not in exclude_cols]

clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=100, random_state=42, class_weight="balanced")
clf.fit(master_panel[feat_cols].values, master_panel["target"].values)

X_live = live_df[feat_cols].values
probs_up = clf.predict_proba(X_live)[:, 1]
preds_up = clf.predict(X_live)

live_df["AI_Direction"] = ["UP" if p == 1 else "DOWN" for p in preds_up]
live_df["AI_Confidence"] = probs_up

orders_df = live_df[live_df["AI_Direction"] == "UP"].sort_values(by="AI_Confidence", ascending=False).head(TOP_K_ASSETS)

# Determine targeted ticker lists for tomorrow
if orders_df.empty:
    tomorrow_positions = ["CASH"]
else:
    tomorrow_positions = list(orders_df.index)
    # Deduct transaction fee adjustments based on allocation shifts
    new_buys = set(tomorrow_positions) - set(last_held_positions)
    sales = set(last_held_positions) - set(tomorrow_positions)
    current_capital -= (current_capital / TOP_K_ASSETS) * (len(new_buys) + len(sales)) * FEES_PER_SIDE

# ── 5. SAVE RECORDS BACK INTO JSON DATABASE ───────────────────────────────────
# Clean duplicates if running manually twice in a single day
if history[-1]["date"] == today_str:
    history[-1] = {"date": today_str, "capital": float(current_capital), "positions": tomorrow_positions}
else:
    history.append({"date": today_str, "capital": float(current_capital), "positions": tomorrow_positions})

with open(HISTORY_FILE, "w") as f:
    json.dump(history, f, indent=2)

print(f"Success. Target Tomorrow: {tomorrow_positions}")
