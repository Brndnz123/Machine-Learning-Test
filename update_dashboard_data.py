#!/usr/bin/env python3
"""
BIST Systematic Portfolio Engine v5
====================================

One code path for:
  1) a purged walk-forward historical backtest, and
  2) a live daily deployment run.

Key v5 corrections:
  - Point-in-time universe support (required for unbiased backtests)
  - Purged walk-forward training: no training label may extend into the
    prediction date
  - Causal features (features use information available before the decision)
  - Covariance-aware portfolio volatility targeting
  - Trade-based ADV participation limits using current capital and current
    holdings
  - Explicit cash / financing accounting
  - Idempotent, atomic live-state persistence
  - Historical peak preserved independently of the rolling trade log
  - Realistic, explicit transaction/slippage/financing assumptions
  - Full out-of-sample portfolio metrics in backtest mode

Expected optional input files:
  universe_history.csv: date,ticker
      Point-in-time membership. Each row says ticker belongs to the eligible
      universe on/after date until the next membership change.

  risk_free_daily.csv: date,annual_rate
      Daily observations of annualized risk-free rate. If absent, the engine
      can use CASH_ANNUAL_RATE as an explicit fallback, but backtests should
      preferably use a time-varying series.

Usage examples:
  python update_dashboard_data_v5.py --mode backtest \
      --universe-history universe_history.csv \
      --risk-free risk_free_daily.csv \
      --start 2023-01-03 --end 2026-09-01

  python update_dashboard_data_v5.py --mode live \
      --universe-history universe_history.csv \
      --risk-free risk_free_daily.csv

Dependencies:
  numpy, pandas, yfinance, scikit-learn
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# =============================================================================
# UNIVERSE / CONFIGURATION
# =============================================================================

# This is the fallback live universe. It is NOT a point-in-time historical
# universe and therefore should not be used as the historical membership set
# for an unbiased backtest.
CURRENT_BIST_UNIVERSE: Tuple[str, ...] = (
    "AKBNK.IS", "GARAN.IS", "HALKB.IS", "ISCTR.IS", "SKBNK.IS", "VAKBN.IS",
    "YKBNK.IS", "ALBRK.IS", "TSKB.IS", "ISMEN.IS", "SAHOL.IS", "KCHOL.IS",
    "AGHOL.IS", "DOHOL.IS", "GSDHO.IS", "SMRTG.IS", "AKSEN.IS", "AYDEM.IS",
    "ENJSA.IS", "EUPWR.IS", "GWIND.IS", "CWENE.IS", "ZOREN.IS", "ODAS.IS",
    "TUPRS.IS", "EREGL.IS", "KRDMD.IS", "BRSAN.IS", "OYAKC.IS", "AKSA.IS",
    "PETKM.IS", "SASA.IS", "GUBRF.IS", "CEMTS.IS", "BUCIM.IS", "AKCNS.IS",
    "CIMSA.IS", "SISE.IS", "ENKAI.IS", "EKGYO.IS", "ISGYO.IS", "TKFEN.IS",
    "FROTO.IS", "TOASO.IS", "OTKAR.IS", "TTRAK.IS", "ASUZU.IS", "ARCLK.IS",
    "VESTL.IS", "VESBE.IS", "ASELS.IS", "GESAN.IS", "GENIL.IS", "GOLTS.IS",
    "KCAER.IS", "KONTR.IS", "KORDS.IS", "ALFAS.IS", "HEKTS.IS", "DOAS.IS",
    "BRYAT.IS", "MAVI.IS", "PENTA.IS", "BIMAS.IS", "MGROS.IS", "SOKM.IS",
    "ULKER.IS", "CCOLA.IS", "TUKAS.IS", "AEFES.IS", "KENT.IS", "YYLGD.IS",
    "TABGD.IS", "TCELL.IS", "TTKOM.IS", "LOGO.IS", "SDTTR.IS", "ASTOR.IS",
    "MIATK.IS", "PGSUS.IS", "TAVHL.IS", "THYAO.IS", "TURSG.IS", "QUAGR.IS",
    "SAYAS.IS", "UFUK.IS", "EGEEN.IS", "ECILC.IS", "YEOTK.IS",
)


@dataclass(frozen=True)
class Config:
    market_ticker: str = "XU100.IS"
    fx_ticker: str = "USDTRY=X"

    data_lookback_years: int = 5
    target_horizon: int = 21
    top_k: int = 5
    min_train_rows: int = 1000
    retrain_every_n_days: int = 1

    vol_target: float = 0.20
    max_gross_leverage: float = 1.25
    max_single_position_weight: float = 0.35
    vol_floor_daily: float = 0.01 / math.sqrt(252.0)
    covariance_lookback: int = 126
    covariance_shrinkage: float = 0.15

    min_adv_try: float = 5_000_000.0
    min_active_day_ratio: float = 0.80
    participation_limit: float = 0.05

    model_weight: float = 0.70
    relative_strength_weight: float = 0.30
    winsorize_alpha: bool = True

    transaction_cost: float = 0.0015
    slippage_cost: float = 0.0005
    financing_spread: float = 0.03

    cash_annual_rate: float = 0.45
    starting_capital: float = 100_000.0

    backtest_start: Optional[str] = None
    backtest_end: Optional[str] = None

    universe_history_path: str = "universe_history.csv"
    risk_free_path: str = "risk_free_daily.csv"
    state_path: str = "state_v5.json"
    trades_path: str = "backtest_trades_v5.csv"
    equity_path: str = "backtest_equity_v5.csv"
    metrics_path: str = "backtest_metrics_v5.json"


# =============================================================================
# INPUT PROVIDERS
# =============================================================================

class UniverseProvider:
    """Point-in-time membership provider.

    File format:
        date,ticker
        2023-01-03,THYAO.IS
        2023-01-03,AKBNK.IS
        2023-04-04,THYAO.IS

    Semantics: a ticker is a member from its latest membership date onward.
    To model exits explicitly, a membership table should include a full
    rebalance snapshot on each effective date (the rows for that date are the
    entire eligible universe), rather than attempting to encode exits as
    individual negative rows.
    """

    def __init__(self, path: str, allow_fallback: bool) -> None:
        self.path = Path(path)
        self.allow_fallback = allow_fallback
        self._table: Optional[pd.DataFrame] = None
        self._snapshots: Dict[pd.Timestamp, set[str]] = {}

        if self.path.exists():
            df = pd.read_csv(self.path)
            required = {"date", "ticker"}
            if not required.issubset(df.columns):
                raise ValueError(f"{path} must contain columns: {sorted(required)}")
            df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
            df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
            df = df.dropna(subset=["date", "ticker"])
            self._table = df.sort_values(["date", "ticker"]).drop_duplicates()
            for date, group in self._table.groupby("date"):
                self._snapshots[pd.Timestamp(date)] = set(group["ticker"])

        elif not allow_fallback:
            raise FileNotFoundError(
                f"Point-in-time universe file not found: {path}. "
                "A historical backtest must supply universe_history.csv."
            )

    @property
    def uses_fallback(self) -> bool:
        return self._table is None

    def members_on(self, date: pd.Timestamp) -> set[str]:
        date = pd.Timestamp(date).normalize()
        if self._table is None:
            return set(CURRENT_BIST_UNIVERSE)

        effective_dates = [d for d in self._snapshots if d <= date]
        if not effective_dates:
            return set()
        latest = max(effective_dates)
        return set(self._snapshots[latest])

    def all_tickers(self) -> set[str]:
        if self._table is None:
            return set(CURRENT_BIST_UNIVERSE)
        return set(self._table["ticker"])


class RiskFreeProvider:
    """Daily risk-free rate provider with explicit fallback support."""

    def __init__(self, path: str, fallback_annual_rate: float) -> None:
        self.fallback_annual_rate = float(fallback_annual_rate)
        self.path = Path(path)
        self.series: Optional[pd.Series] = None

        if self.path.exists():
            df = pd.read_csv(self.path)
            required = {"date", "annual_rate"}
            if not required.issubset(df.columns):
                raise ValueError(f"{path} must contain columns: {sorted(required)}")
            df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
            df["annual_rate"] = pd.to_numeric(df["annual_rate"], errors="raise")
            df = df.drop_duplicates("date", keep="last").sort_values("date")
            self.series = df.set_index("date")["annual_rate"].astype(float)

    @property
    def uses_fallback(self) -> bool:
        return self.series is None

    def annual_rate_on(self, date: pd.Timestamp) -> float:
        date = pd.Timestamp(date).normalize()
        if self.series is None:
            return self.fallback_annual_rate
        s = self.series.loc[:date]
        if s.empty:
            return self.fallback_annual_rate
        return float(s.iloc[-1])

    def daily_rate_on(self, date: pd.Timestamp) -> float:
        annual = self.annual_rate_on(date)
        return (1.0 + annual) ** (1.0 / 252.0) - 1.0


# =============================================================================
# DATA / FEATURES
# =============================================================================

class MarketData:
    def __init__(self, raw: pd.DataFrame, universe_provider: UniverseProvider, cfg: Config) -> None:
        self.raw = raw
        self.universe_provider = universe_provider
        self.cfg = cfg
        self.market_close = self._series(cfg.market_ticker, "Close")
        self.fx_close = self._series(cfg.fx_ticker, "Close")
        self.market_ret_1d = self.market_close.pct_change()
        self.returns = pd.DataFrame()
        self.volume_value = pd.DataFrame()

    def _series(self, ticker: str, field: str) -> pd.Series:
        try:
            s = self.raw[ticker][field].copy()
        except KeyError as exc:
            raise RuntimeError(f"Required data missing for {ticker}/{field}") from exc
        return s.dropna().sort_index()

    def stock_frame(self, ticker: str) -> pd.DataFrame:
        try:
            return self.raw[ticker].copy()
        except KeyError:
            return pd.DataFrame()

    def validate(self) -> None:
        if self.market_close.empty:
            raise RuntimeError("Benchmark data is empty.")
        if self.fx_close.empty:
            raise RuntimeError("FX data is empty.")

        bad_market = self.market_close[(self.market_close <= 0) | ~np.isfinite(self.market_close)]
        if not bad_market.empty:
            raise RuntimeError("Benchmark contains non-positive/non-finite prices.")

    def build_panel(self) -> pd.DataFrame:
        panel: List[pd.DataFrame] = []
        returns: Dict[str, pd.Series] = {}
        volume_value: Dict[str, pd.Series] = {}

        market_features = pd.DataFrame(index=self.market_close.index)
        market_features["mkt_ret_5d"] = self.market_close.pct_change(5)
        market_features["mkt_ret_21d"] = self.market_close.pct_change(21)
        market_features["mkt_vol_21d"] = self.market_ret_1d.rolling(21).std()
        market_features["mkt_vol_63d"] = self.market_ret_1d.rolling(63).std()
        market_features["mkt_ma_10_50"] = (
            self.market_close.rolling(10).mean() / self.market_close.rolling(50).mean()
        )
        market_features["mkt_ma_50_200"] = (
            self.market_close.rolling(50).mean() / self.market_close.rolling(200).mean()
        )
        market_features["fx_ret_5d"] = self.fx_close.pct_change(5).reindex(self.market_close.index).ffill()
        market_features["fx_ret_21d"] = self.fx_close.pct_change(21).reindex(self.market_close.index).ffill()
        market_features = market_features.shift(1)

        tickers = sorted(self.universe_provider.all_tickers())
        for ticker in tickers:
            df = self.stock_frame(ticker)
            if df.empty or "Close" not in df.columns or "Volume" not in df.columns:
                continue
            df = df.dropna(subset=["Close"])
            if df.empty:
                continue

            close = pd.to_numeric(df["Close"], errors="coerce")
            volume = pd.to_numeric(df["Volume"], errors="coerce")
            valid = close > 0
            close = close.where(valid)
            volume = volume.where(volume >= 0)
            ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
            returns[ticker] = ret
            volume_value[ticker] = (close * volume)

            feats = pd.DataFrame(index=close.index)
            for lag in (1, 5, 10, 21, 63):
                feats[f"ret_{lag}d"] = close.pct_change(lag).shift(1)
            for lag in (5, 21, 63):
                feats[f"vol_{lag}d"] = ret.rolling(lag).std().shift(1)

            feats["skew_21d"] = ret.rolling(21).skew().shift(1)
            feats["kurt_21d"] = ret.rolling(21).kurt().shift(1)
            feats["intraday_range"] = ((df["High"] - df["Low"]) / close).shift(1)

            vol_std = volume.rolling(21).std().replace(0, np.nan)
            feats["volume_z"] = (
                (volume - volume.rolling(21).mean()) / vol_std
            ).shift(1)
            feats["trend_ma_ratio"] = (
                close.rolling(10).mean() / close.rolling(50).mean()
            ).shift(1)

            market_ret_21 = self.market_close.pct_change(21)
            aligned_market = market_ret_21.reindex(close.index).ffill()
            feats["relative_strength_21d"] = (
                close.pct_change(21) - aligned_market
            ).shift(1)

            # Previous-day information only. The value is TRY traded value,
            # using adjusted close consistently with the downloaded price series.
            feats["adv_63d_try"] = volume_value[ticker].rolling(63).mean().shift(1)

            feats = feats.join(market_features, how="left")
            feats["target"] = close.pct_change(self.cfg.target_horizon).shift(-self.cfg.target_horizon)
            feats["ticker"] = ticker

            # Label end is the actual future observation date corresponding to
            # the target. This is used to purge training rows that would overlap
            # the prediction date.
            idx = pd.DatetimeIndex(feats.index)
            label_end = pd.Series(index=idx, dtype="datetime64[ns]")
            pos = np.arange(len(idx)) + self.cfg.target_horizon
            good = pos < len(idx)
            label_end.loc[idx[good]] = idx[pos[good]]
            feats["label_end"] = label_end.values

            panel.append(feats)

        if not panel:
            raise RuntimeError("No stock feature panels were constructed.")

        self.returns = pd.DataFrame(returns).sort_index()
        self.volume_value = pd.DataFrame(volume_value).sort_index()
        master = pd.concat(panel).sort_index()

        # Restrict every observation to its point-in-time eligible universe.
        # This is essential for historical backtesting; for live mode the
        # provider may legitimately use the current fallback universe.
        keep = []
        for date, group in master.groupby(level=0, sort=False):
            members = self.universe_provider.members_on(pd.Timestamp(date))
            keep.append(group[group["ticker"].isin(members)])
        master = pd.concat(keep).sort_index() if keep else master.iloc[0:0]
        return master


# =============================================================================
# MODEL
# =============================================================================

FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d",
    "vol_5d", "vol_21d", "vol_63d", "skew_21d", "kurt_21d",
    "intraday_range", "volume_z", "trend_ma_ratio",
    "relative_strength_21d", "mkt_ret_5d", "mkt_ret_21d",
    "mkt_vol_21d", "mkt_vol_63d", "mkt_ma_10_50", "mkt_ma_50_200",
    "fx_ret_5d", "fx_ret_21d",
]


def make_model() -> Pipeline:
    ensemble = VotingRegressor(
        estimators=[
            (
                "gbm",
                HistGradientBoostingRegressor(
                    max_depth=3,
                    max_iter=150,
                    learning_rate=0.03,
                    l2_regularization=0.5,
                    random_state=42,
                ),
            ),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=6,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ],
        weights=[0.6, 0.4],
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("ensemble", ensemble),
        ]
    )


class SignalEngine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.model: Optional[Pipeline] = None
        self.last_fit_date: Optional[pd.Timestamp] = None

    def fit_if_needed(self, panel: pd.DataFrame, decision_date: pd.Timestamp) -> bool:
        if (
            self.model is not None
            and self.last_fit_date is not None
            and (decision_date - self.last_fit_date).days < self.cfg.retrain_every_n_days
        ):
            return False

        # Purge by actual label end: all training targets must be completely
        # finished strictly before the prediction/decision date.
        train = panel[
            (panel.index < decision_date)
            & (pd.to_datetime(panel["label_end"]) < decision_date)
            & panel["target"].notna()
        ].copy()

        if len(train) < self.cfg.min_train_rows:
            raise RuntimeError(
                f"Insufficient purged training rows at {decision_date.date()}: "
                f"{len(train)} < {self.cfg.min_train_rows}."
            )

        x = train[FEATURE_COLUMNS]
        y = train["target"].astype(float)
        model = make_model()
        model.fit(x.values, y.values)
        self.model = model
        self.last_fit_date = decision_date
        return True

    @staticmethod
    def zscore(values: pd.Series) -> pd.Series:
        std = values.std(ddof=0)
        if not np.isfinite(std) or std <= 0:
            return pd.Series(0.0, index=values.index)
        return (values - values.mean()) / std

    def predict(self, day_panel: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Signal model is not fitted.")
        x = day_panel[FEATURE_COLUMNS]
        pred = pd.Series(self.model.predict(x.values), index=day_panel["ticker"].astype(str))
        if self.cfg.winsorize_alpha and len(pred) >= 10:
            lo, hi = pred.quantile(0.02), pred.quantile(0.98)
            pred = pred.clip(lo, hi)
        rs = pd.Series(day_panel["relative_strength_21d"].values, index=day_panel["ticker"])
        score = self.cfg.model_weight * self.zscore(pred) + self.cfg.relative_strength_weight * self.zscore(rs)
        return score.sort_values(ascending=False)


# =============================================================================
# PORTFOLIO / RISK / EXECUTION
# =============================================================================

class PortfolioEngine:
    def __init__(self, cfg: Config, rf: RiskFreeProvider) -> None:
        self.cfg = cfg
        self.rf = rf

    @staticmethod
    def _nearest_psd(cov: np.ndarray, eigen_floor: float = 1e-10) -> np.ndarray:
        cov = (cov + cov.T) / 2.0
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, eigen_floor)
        return (vecs * vals) @ vecs.T

    def covariance(self, returns: pd.DataFrame, date: pd.Timestamp, tickers: Sequence[str]) -> pd.DataFrame:
        end = pd.Timestamp(date) - pd.Timedelta(days=1)
        window = returns.loc[:end, list(tickers)].tail(self.cfg.covariance_lookback)
        window = window.dropna(axis=1, how="all")
        if window.shape[0] < 30:
            # Diagonal fallback: still causal, but conservative relative to
            # unstable short-history correlations.
            vols = window.std().replace([np.inf, -np.inf], np.nan).fillna(self.cfg.vol_floor_daily)
            cov = np.diag(np.maximum(vols.values, self.cfg.vol_floor_daily) ** 2)
            return pd.DataFrame(cov, index=vols.index, columns=vols.index)

        cov = window.cov().fillna(0.0)
        # Simple shrinkage toward diagonal for stability.
        diag = pd.DataFrame(np.diag(np.diag(cov.values)), index=cov.index, columns=cov.columns)
        lam = float(np.clip(self.cfg.covariance_shrinkage, 0.0, 1.0))
        shrunk = (1.0 - lam) * cov + lam * diag
        shrunk_values = self._nearest_psd(shrunk.values)
        return pd.DataFrame(shrunk_values, index=shrunk.index, columns=shrunk.columns)

    def desired_weights(
        self,
        decision_date: pd.Timestamp,
        selected: Sequence[str],
        returns: pd.DataFrame,
    ) -> Tuple[Dict[str, float], float]:
        tickers = list(selected)
        if not tickers:
            return {}, 0.0

        cov = self.covariance(returns, decision_date, tickers)
        vol = np.sqrt(np.maximum(np.diag(cov.values), 0.0))
        vol = np.maximum(vol, self.cfg.vol_floor_daily)

        inv_vol = 1.0 / vol
        raw = inv_vol / inv_vol.sum()
        port_daily_var = float(raw @ cov.loc[tickers, tickers].values @ raw)
        port_ann_vol = math.sqrt(max(port_daily_var, 0.0) * 252.0)

        scale = self.cfg.vol_target / port_ann_vol if port_ann_vol > 0 else 1.0
        scale = min(scale, self.cfg.max_gross_leverage)
        weights = raw * scale

        # The cap deliberately leaves residual exposure as cash rather than
        # renormalizing, because renormalizing could immediately recreate the
        # cap breach and distort the volatility target.
        weights = np.minimum(weights, self.cfg.max_single_position_weight)
        out = {ticker: float(w) for ticker, w in zip(tickers, weights) if w > 0}
        return out, port_ann_vol

    def apply_participation_caps(
        self,
        target: Mapping[str, float],
        current: Mapping[str, float],
        capital: float,
        adv: Mapping[str, float],
    ) -> Dict[str, float]:
        """Limit *trade size*, not position size, to participation × ADV."""
        universe = set(target) | set(current)
        out: Dict[str, float] = {}

        if capital <= 0:
            raise ValueError("Capital must be positive for portfolio sizing.")

        for asset in universe:
            if asset == "CASH":
                continue
            old_w = float(current.get(asset, 0.0))
            new_w = float(target.get(asset, 0.0))
            adv_value = float(adv.get(asset, np.nan))

            if np.isfinite(adv_value) and adv_value > 0:
                max_delta_w = (adv_value * self.cfg.participation_limit) / capital
                delta = np.clip(new_w - old_w, -max_delta_w, max_delta_w)
                executed_w = old_w + delta
            else:
                # Unknown liquidity => do not initiate/increase a trade; allow
                # a legacy holding to remain until liquidity data returns.
                executed_w = min(old_w, new_w) if new_w < old_w else old_w

            if abs(executed_w) > 1e-12:
                out[asset] = float(executed_w)

        # Normalize numerical noise. Cash is represented explicitly elsewhere.
        return out

    def apply_execution_cost(
        self,
        capital: float,
        current: Mapping[str, float],
        target: Mapping[str, float],
    ) -> Tuple[float, float]:
        assets = set(current) | set(target)
        turnover = sum(abs(float(target.get(a, 0.0)) - float(current.get(a, 0.0))) for a in assets if a != "CASH")
        total_cost = turnover * (self.cfg.transaction_cost + self.cfg.slippage_cost)
        return capital * max(0.0, 1.0 - total_cost), turnover

    def mark_to_market(
        self,
        capital: float,
        weights: Mapping[str, float],
        returns_row: Mapping[str, float],
        date: pd.Timestamp,
    ) -> Tuple[float, float, float]:
        risky = 0.0
        for asset, weight in weights.items():
            if asset == "CASH":
                continue
            r = float(returns_row.get(asset, 0.0))
            if not np.isfinite(r):
                r = 0.0
            risky += float(weight) * r

        gross = sum(abs(float(w)) for a, w in weights.items() if a != "CASH")
        cash_weight = 1.0 - gross
        rf_daily = self.rf.daily_rate_on(date)

        # Positive residual cash earns the daily risk-free rate. Negative cash
        # pays the same base rate plus a financing spread.
        cash_pnl = cash_weight * rf_daily if cash_weight >= 0 else cash_weight * (rf_daily + self.cfg.financing_spread)
        total_return = risky + cash_pnl
        new_capital = capital * (1.0 + total_return)
        return new_capital, total_return, cash_weight


# =============================================================================
# STATE / METRICS
# =============================================================================

@dataclass
class EngineState:
    date: str
    capital: float
    benchmark_capital: float
    peak_capital: float
    positions: Dict[str, float]
    total_turnover: float
    circuit_breaker: bool = False


def atomic_json_write(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def load_state(path: str, starting_capital: float) -> EngineState:
    p = Path(path)
    if not p.exists():
        return EngineState(
            date="",
            capital=float(starting_capital),
            benchmark_capital=float(starting_capital),
            peak_capital=float(starting_capital),
            positions={"CASH": 1.0},
            total_turnover=0.0,
            circuit_breaker=False,
        )

    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"State file is unreadable/corrupt: {p}. Refusing to reset capital.") from exc

    required = {"date", "capital", "benchmark_capital", "peak_capital", "positions", "total_turnover"}
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"State file missing required fields: {sorted(missing)}")

    positions = {str(k): float(v) for k, v in payload["positions"].items()}
    capital = float(payload["capital"])
    peak = float(payload["peak_capital"])
    if not np.isfinite(capital) or capital <= 0 or not np.isfinite(peak) or peak <= 0:
        raise RuntimeError("State file contains invalid capital/peak values.")

    return EngineState(
        date=str(payload["date"]),
        capital=capital,
        benchmark_capital=float(payload["benchmark_capital"]),
        peak_capital=peak,
        positions=positions,
        total_turnover=float(payload["total_turnover"]),
        circuit_breaker=bool(payload.get("circuit_breaker", False)),
    )


def save_state(path: str, state: EngineState) -> None:
    atomic_json_write(Path(path), asdict(state))


def performance_metrics(equity: pd.DataFrame, rf: RiskFreeProvider) -> Dict[str, float]:
    if equity.empty or len(equity) < 2:
        return {}

    eq = equity.copy().sort_values("date")
    eq["date"] = pd.to_datetime(eq["date"])
    ret = eq["capital"].pct_change().dropna()
    rf_daily = pd.Series(
        [rf.daily_rate_on(pd.Timestamp(d)) for d in eq.loc[ret.index, "date"]],
        index=ret.index,
        dtype=float,
    )
    excess = ret - rf_daily
    ann_vol = float(ret.std(ddof=1) * math.sqrt(252.0)) if len(ret) > 1 else 0.0
    sharpe = float(excess.mean() / excess.std(ddof=1) * math.sqrt(252.0)) if excess.std(ddof=1) > 0 else 0.0
    years = max((eq["date"].iloc[-1] - eq["date"].iloc[0]).days / 365.25, 1 / 365.25)
    cagr = float((eq["capital"].iloc[-1] / eq["capital"].iloc[0]) ** (1.0 / years) - 1.0)
    dd = eq["capital"] / eq["capital"].cummax() - 1.0
    max_dd = float(dd.min())
    benchmark_return = float(eq["benchmark_capital"].iloc[-1] / eq["benchmark_capital"].iloc[0] - 1.0)
    total_turnover = float(eq["turnover"].sum()) if "turnover" in eq else 0.0
    return {
        "start": eq["date"].iloc[0].strftime("%Y-%m-%d"),
        "end": eq["date"].iloc[-1].strftime("%Y-%m-%d"),
        "ending_capital": float(eq["capital"].iloc[-1]),
        "cagr": cagr,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "benchmark_total_return": benchmark_return,
        "average_daily_turnover": float(total_turnover / max(len(ret), 1)),
        "total_turnover": total_turnover,
    }


# =============================================================================
# ENGINE
# =============================================================================

class BISTV5Engine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def download_data(self, universe: UniverseProvider, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError(
                "yfinance is required for data acquisition. Install dependencies with: pip install -r requirements-v5.txt"
            ) from exc

        tickers = sorted(universe.all_tickers() | {self.cfg.market_ticker, self.cfg.fx_ticker})
        print(f"Downloading {len(tickers)} tickers from yFinance...")
        raw = yf.download(
            tickers=tickers,
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
        if raw is None or raw.empty:
            raise RuntimeError("yFinance returned no data.")
        if not isinstance(raw.columns, pd.MultiIndex):
            raise RuntimeError("Unexpected yFinance column layout; expected multi-ticker data.")
        raw = raw.sort_index()
        return raw

    def prepare(self, mode: str) -> Tuple[MarketData, UniverseProvider, RiskFreeProvider, pd.Timestamp, pd.Timestamp]:
        fallback_allowed = mode == "live"
        universe = UniverseProvider(self.cfg.universe_history_path, allow_fallback=fallback_allowed)
        rf = RiskFreeProvider(self.cfg.risk_free_path, self.cfg.cash_annual_rate)

        now = pd.Timestamp(datetime.now(timezone.utc).date())
        if mode == "backtest":
            start = pd.Timestamp(self.cfg.backtest_start) if self.cfg.backtest_start else now - pd.DateOffset(years=3)
            end = pd.Timestamp(self.cfg.backtest_end) if self.cfg.backtest_end else now
        else:
            start = now - pd.DateOffset(years=self.cfg.data_lookback_years)
            end = now

        # Add warm-up before the requested start for feature/covariance history.
        download_start = start - pd.DateOffset(years=self.cfg.data_lookback_years)
        raw = self.download_data(universe, download_start, end)
        market = MarketData(raw, universe, self.cfg)
        market.validate()
        return market, universe, rf, start.normalize(), end.normalize()

    @staticmethod
    def active_ratio(returns: pd.DataFrame, end_date: pd.Timestamp, tickers: Iterable[str], lookback: int = 252) -> pd.Series:
        hist = returns.loc[: end_date - pd.Timedelta(days=1)].tail(lookback)
        if hist.empty:
            return pd.Series(dtype=float)
        return hist[list(tickers)].notna().mean()

    def create_target(
        self,
        market: MarketData,
        panel: pd.DataFrame,
        signal: SignalEngine,
        portfolio: PortfolioEngine,
        decision_date: pd.Timestamp,
        current_positions: Mapping[str, float],
        capital: float,
    ) -> Tuple[Dict[str, float], Dict[str, object]]:
        day = panel.loc[panel.index == decision_date].copy()
        if day.empty:
            return {}, {"selected": [], "reason": "no_panel"}

        active = self.active_ratio(market.returns, decision_date, day["ticker"].tolist())
        liquid = day[
            (day["adv_63d_try"] >= self.cfg.min_adv_try)
            & day["ticker"].isin(active[active >= self.cfg.min_active_day_ratio].index)
        ].copy()
        if liquid.empty:
            return {}, {"selected": [], "reason": "no_liquid_assets"}

        signal.fit_if_needed(panel, decision_date)
        scores = signal.predict(liquid)
        selected = list(scores.head(self.cfg.top_k).index)
        desired, model_port_vol = portfolio.desired_weights(decision_date, selected, market.returns)

        adv = dict(zip(liquid["ticker"], liquid["adv_63d_try"].astype(float)))
        executed = portfolio.apply_participation_caps(desired, current_positions, capital, adv)

        # Always permit explicit flattening of positions whose target is zero,
        # subject to participation. Residual current positions can therefore
        # remain temporarily when liquidity throttles exits.
        detail = {
            "selected": selected,
            "desired_weights": desired,
            "executed_weights": executed,
            "model_portfolio_annual_vol_before_caps": model_port_vol,
            "signal_scores": {str(k): float(v) for k, v in scores.head(self.cfg.top_k).items()},
            "liquid_count": int(len(liquid)),
        }
        return executed, detail

    def run_backtest(self) -> pd.DataFrame:
        market, _, rf, start, end = self.prepare("backtest")
        panel = market.build_panel()
        dates = sorted(pd.DatetimeIndex(panel.index.unique()))
        dates = [d for d in dates if start <= d <= end]
        if len(dates) < 2:
            raise RuntimeError("Backtest window contains too few observations.")

        signal = SignalEngine(self.cfg)
        portfolio = PortfolioEngine(self.cfg, rf)
        capital = float(self.cfg.starting_capital)
        benchmark = float(self.cfg.starting_capital)
        positions: Dict[str, float] = {}
        peak = capital
        rows: List[dict] = []
        total_turnover = 0.0

        # We need one prior day to mark the portfolio before the first rebalance.
        all_dates = sorted(pd.DatetimeIndex(panel.index.unique()))
        eligible_indices = [i for i, d in enumerate(all_dates) if d >= start and d <= end]
        if not eligible_indices or eligible_indices[0] == 0:
            first_decision_idx = max(1, eligible_indices[0] if eligible_indices else 1)
        else:
            first_decision_idx = eligible_indices[0]

        # Initialize in cash through the first decision date.
        prev_date = all_dates[first_decision_idx - 1]

        for i in range(first_decision_idx, len(all_dates)):
            date = all_dates[i]
            if date > end:
                break

            # Mark yesterday's portfolio from prev_date -> date.
            # Since positions were set at the previous close, today's return is
            # fully out-of-sample relative to the current decision.
            if positions:
                row = market.returns.reindex([date]).iloc[0].to_dict() if date in market.returns.index else {}
                capital, daily_return, cash_weight = portfolio.mark_to_market(capital, positions, row, date)
            else:
                rf_daily = rf.daily_rate_on(date)
                capital *= (1.0 + rf_daily)
                daily_return = rf_daily
                cash_weight = 1.0

            if date in market.market_ret_1d.index and np.isfinite(market.market_ret_1d.loc[date]):
                benchmark *= (1.0 + float(market.market_ret_1d.loc[date]))

            peak = max(peak, capital)
            current_dd = capital / peak - 1.0

            current = dict(positions)
            # Backtest circuit breaker threshold deliberately conservative and
            # configurable through a simple constant here. A triggered breaker
            # sets the *next* target to cash; it does not retroactively prevent
            # today's return.
            if current_dd <= -0.30:
                target = {}
                detail = {"selected": [], "reason": "30pct_drawdown_circuit_breaker"}
            else:
                target, detail = self.create_target(
                    market, panel, signal, portfolio, date, current, capital
                )

            after_cost, turnover = portfolio.apply_execution_cost(capital, current, target)
            cost_drag = capital - after_cost
            capital = after_cost
            total_turnover += turnover
            positions = dict(target)
            positions["CASH"] = 1.0 - sum(w for a, w in positions.items() if a != "CASH")

            rows.append(
                {
                    "date": date,
                    "capital": capital,
                    "benchmark_capital": benchmark,
                    "daily_return_before_cost": daily_return,
                    "cost_drag": cost_drag / max(capital + cost_drag, 1e-12),
                    "turnover": turnover,
                    "cash_weight": positions.get("CASH", 0.0),
                    "drawdown": current_dd,
                    "selected": json.dumps(detail.get("selected", [])),
                    "weights": json.dumps({k: float(v) for k, v in positions.items()}, sort_keys=True),
                    "reason": detail.get("reason", "rebalance"),
                }
            )
            prev_date = date

        equity = pd.DataFrame(rows)
        if equity.empty:
            raise RuntimeError("Backtest produced no rows.")

        metrics = performance_metrics(equity, rf)
        metrics["universe_point_in_time"] = not UniverseProvider(self.cfg.universe_history_path, allow_fallback=True).uses_fallback
        metrics["risk_free_time_varying"] = not rf.uses_fallback
        metrics["notes"] = (
            "Returns are out-of-sample one-period portfolio returns. Model training is purged so labels "
            "whose end date reaches the decision date are excluded. Execution is modeled at the decision "
            "close using causal features from the prior session."
        )

        equity.to_csv(self.cfg.equity_path, index=False)
        Path(self.cfg.metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("\n=== V5 WALK-FORWARD BACKTEST ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print(f"Saved equity:  {self.cfg.equity_path}")
        print(f"Saved metrics: {self.cfg.metrics_path}")
        return equity

    def run_live(self) -> Dict[str, object]:
        market, _, rf, _, end = self.prepare("live")
        panel = market.build_panel()
        if panel.empty:
            raise RuntimeError("Live panel is empty.")
        latest_date = max(pd.DatetimeIndex(panel.index.unique()))
        if latest_date > end:
            latest_date = end

        state = load_state(self.cfg.state_path, self.cfg.starting_capital)
        latest_str = latest_date.strftime("%Y-%m-%d")
        if state.date == latest_str:
            print(f"Already synchronized for {latest_str}; no-op.")
            return asdict(state)

        # Mark current holdings through latest market date.
        if state.date:
            prior_date = pd.Timestamp(state.date)
            if latest_date <= prior_date:
                raise RuntimeError(
                    f"State date {state.date} is not before latest market date {latest_str}."
                )
            for date in market.returns.index:
                date = pd.Timestamp(date)
                if prior_date < date <= latest_date:
                    state.capital, _, _ = PortfolioEngine(self.cfg, rf).mark_to_market(
                        state.capital,
                        state.positions,
                        market.returns.loc[date].to_dict(),
                        date,
                    )
        else:
            # Initial deployment remains entirely in cash until the first signal.
            state.capital *= 1.0

        portfolio = PortfolioEngine(self.cfg, rf)
        signal = SignalEngine(self.cfg)
        target, detail = self.create_target(
            market, panel, signal, portfolio, latest_date, state.positions, state.capital
        )

        after_cost, turnover = portfolio.apply_execution_cost(state.capital, state.positions, target)
        state.capital = after_cost
        state.positions = dict(target)
        state.positions["CASH"] = 1.0 - sum(w for a, w in state.positions.items() if a != "CASH")
        state.peak_capital = max(state.peak_capital, state.capital)
        state.total_turnover += turnover
        state.date = latest_str
        state.circuit_breaker = False

        # Idempotent, atomic state commit.
        save_state(self.cfg.state_path, state)

        result = {
            **asdict(state),
            "selected": detail.get("selected", []),
            "desired_weights": detail.get("desired_weights", {}),
            "executed_weights": detail.get("executed_weights", {}),
            "signal_scores": detail.get("signal_scores", {}),
        }
        print("\n=== V5 LIVE ENGINE ===")
        print(json.dumps(result, indent=2, default=str))
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BIST v5 walk-forward backtester/live engine")
    parser.add_argument("--mode", choices=("backtest", "live"), default="backtest")
    parser.add_argument("--start", dest="start", default=None)
    parser.add_argument("--end", dest="end", default=None)
    parser.add_argument("--universe-history", default="universe_history.csv")
    parser.add_argument("--risk-free", default="risk_free_daily.csv")
    parser.add_argument("--state", default="state_v5.json")
    parser.add_argument("--equity", default="backtest_equity_v5.csv")
    parser.add_argument("--metrics", default="backtest_metrics_v5.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        backtest_start=args.start,
        backtest_end=args.end,
        universe_history_path=args.universe_history,
        risk_free_path=args.risk_free,
        state_path=args.state,
        equity_path=args.equity,
        metrics_path=args.metrics,
    )

    engine = BISTV5Engine(cfg)
    if args.mode == "backtest":
        engine.run_backtest()
    else:
        engine.run_live()


if __name__ == "__main__":
    main()
