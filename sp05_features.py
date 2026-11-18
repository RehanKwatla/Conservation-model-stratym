"""SP-05 Conservative Risk-Aware Policy - shared constants and causal feature engineering.

Schema family rule: the first 30 sequence features and the first 8 static features are identical to the other
policies (shared contract). SP-05 adds 13 risk-oriented sequence features (produced by `RiskFeatureAdapter`
from OHLCV + benchmark, passed as `risk_features` or appended to `sequence`) and 4 portfolio-level static
inputs (portfolio drawdown, gross exposure, position size, recent loss; all default to 0).
Every feature at bar t uses data up to and including the close of bar t only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Common action space (shared by all five specialised policies)
# ----------------------------------------------------------------------------
HOLD, BUY, SELL, EXIT = 0, 1, 2, 3
ACTION_NAMES = ["HOLD", "BUY", "SELL", "EXIT"]
N_ACTIONS = 4

# Position state: +1 long, 0 flat, -1 short. One-hot order below is part of the schema.
POSITION_ORDER = (0, 1, -1)                      # one-hot index -> position value
POSITION_NAMES = ["pos_flat", "pos_long", "pos_short"]

# Trader-profile features, each in [0, 1].
PROFILE_FEATURES = ["risk_tolerance", "patience", "aggressiveness"]
DEFAULT_PROFILE = (0.5, 0.5, 0.5)

STATIC_FEATURE_NAMES = POSITION_NAMES + PROFILE_FEATURES   # shared prefix: 3 + 3 = 6 dims

# Sequential market features: exact column order of the model input (30 dims).
BASE_FEATURE_COLUMNS = [
    # returns
    "ret_1", "logret_1", "ret_5", "ret_10", "ret_20", "ret_60",
    # momentum oscillators
    "rsi_14", "macd_pct", "macd_signal_pct", "macd_hist_pct",
    # moving-average relationships
    "close_ema20", "close_ema50", "ema20_ema50", "close_sma200", "sma50_sma200",
    # trend strength
    "adx_14", "di_diff",
    # volatility
    "atr_pct", "rv_20", "rv_60",
    # volume
    "vol_chg_1", "vol_ratio_20", "vol_z_60",
    # candle shape
    "hl_range", "gap_open", "close_loc",
    # relative strength vs benchmark and market context
    "rel_strength_20", "rel_strength_60", "bench_ret_20", "bench_rv_20",
]
# SP-05 additions (appended after the 30 shared features; see RiskFeatureAdapter)
EXTRA_FEATURE_COLUMNS = [
    "atr_pct_pctile_252", "rv20_pctile_252", "dd_from_high_60", "dd_from_high_252", "maxdd_60",
    "gap_abs_atr", "abnormal_ret_z", "range_z", "vol_anomaly_z", "bb_width_20", "down_vol_20",
    "bench_dd_60", "bench_rv20_pctile_252",
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS
N_BASE_FEATURES = len(BASE_FEATURE_COLUMNS)
N_FEATURES = len(FEATURE_COLUMNS)
N_BASE_STATIC = len(STATIC_FEATURE_NAMES)
SHARED_STATIC_NAMES = list(STATIC_FEATURE_NAMES) + ["unrealized_pnl_scaled", "holding_bars_scaled"]   # identical to SP-02/03/04 (8 dims)
EXTRA_STATIC_NAMES = ["unrealized_pnl_scaled", "holding_bars_scaled",
                      "portfolio_drawdown_scaled", "gross_exposure_scaled", "position_size_scaled", "recent_loss_scaled"]
STATIC_FEATURE_NAMES = STATIC_FEATURE_NAMES + EXTRA_STATIC_NAMES
N_STATIC = len(STATIC_FEATURE_NAMES)
N_SHARED_STATIC = len(SHARED_STATIC_NAMES)
PNL_SCALE, PNL_CLIP = 0.10, 3.0       # unrealized_pnl_scaled = clip(pnl / 0.10, -3, 3)
HOLD_SCALE, HOLD_CAP = 20.0, 60       # holding_bars_scaled   = min(bars, 60) / 20
PDD_SCALE, PDD_CLIP = 0.10, 3.0       # portfolio_drawdown_scaled = clip(drawdown / 0.10, 0, 3)  (drawdown >= 0: below the trailing high)
GROSS_CLIP = 2.0                      # gross_exposure_scaled = clip(gross / equity, 0, 2)
SIZE_SCALE, SIZE_CLIP = 0.20, 3.0     # position_size_scaled = clip(size / 0.20, 0, 3)  (0 when flat)
LOSS_SCALE, LOSS_CLIP = 0.05, 3.0     # recent_loss_scaled = clip(loss / 0.05, 0, 3)  (loss >= 0: equity fall over ~20 bars)
RISK_MIN_HISTORY = 290
WARMUP_BARS = 290          # percentile ranks over 252 bars of a 20-bar volatility need ~272 bars
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


# ----------------------------------------------------------------------------
# Action validity and position transitions
# ----------------------------------------------------------------------------
def valid_action_mask(position, allow_short: bool = True) -> np.ndarray:
    """Boolean (B, 4) mask of valid actions given the current position.

    flat  -> HOLD, BUY (open long), SELL (open short, only if allow_short)
    long  -> HOLD, EXIT
    short -> HOLD, EXIT
    """
    pos = np.asarray(position).astype(int).reshape(-1)
    mask = np.zeros((pos.size, N_ACTIONS), dtype=bool)
    mask[:, HOLD] = True
    flat = pos == 0
    mask[flat, BUY] = True
    mask[flat, SELL] = bool(allow_short)
    mask[~flat, EXIT] = True
    return mask


def apply_action(position, action) -> np.ndarray:
    """Return the new position after executing (already validated) actions."""
    pos = np.asarray(position).astype(int).reshape(-1)
    act = np.asarray(action).astype(int).reshape(-1)
    new = pos.copy()
    new[(act == BUY) & (pos == 0)] = 1
    new[(act == SELL) & (pos == 0)] = -1
    new[(act == EXIT) & (pos != 0)] = 0
    return new


def profile_array(profile, batch_size: int) -> np.ndarray:
    """Normalise a profile (dict, sequence, (B,3) array or None) to float32 (B, 3)."""
    if profile is None:
        profile = DEFAULT_PROFILE
    if isinstance(profile, dict):
        profile = [profile.get(k, d) for k, d in zip(PROFILE_FEATURES, DEFAULT_PROFILE)]
    arr = np.asarray(profile, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.tile(arr[None, :], (batch_size, 1))
    if arr.shape != (batch_size, len(PROFILE_FEATURES)):
        raise ValueError(f"profile must have shape (3,) or ({batch_size}, 3); got {arr.shape}")
    if np.any(arr < 0) or np.any(arr > 1) or not np.all(np.isfinite(arr)):
        raise ValueError("profile values must be finite and within [0, 1]")
    return arr


def encode_static(position, profile, unrealized_pnl=None, holding_bars=None, portfolio_drawdown=None,
                  gross_exposure=None, position_size=None, recent_loss=None) -> np.ndarray:
    """Static input (B, 12): [flat, long, short] + [risk, patience, aggressiveness] + [pnl, hold] (shared 8 dims)
    + [portfolio_drawdown, gross_exposure, position_size, recent_loss] (SP-05 extras).

    unrealized_pnl     : position-signed fractional P&L since entry (ignored when flat)
    holding_bars       : bars held, entry bar = 1 (ignored when flat)
    portfolio_drawdown : fraction the portfolio equity is below its TRAILING ~120-bar high (0.04 = 4% below); scalar or per state
    gross_exposure     : gross position notional / equity (portfolio level); scalar or per state
    position_size      : this position's notional / equity (ignored when flat)
    recent_loss        : fall of portfolio equity over about the last 20 bars as a fraction (0 if equity rose)
    """
    pos = np.asarray(position).astype(int).reshape(-1)
    if not np.all(np.isin(pos, POSITION_ORDER)):
        raise ValueError("position must be in {-1, 0, +1}")
    n = pos.size
    onehot = np.stack([(pos == p) for p in POSITION_ORDER], axis=1).astype(np.float32)

    def _vec(v, name):
        if v is None:
            return np.zeros(n, dtype=np.float32)
        a = np.asarray(v, dtype=np.float32).reshape(-1)
        if a.size == 1 and n > 1:
            a = np.repeat(a, n)
        if a.size != n or not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be a finite scalar or have one entry per state")
        return a

    flat = pos == 0
    pnl = np.where(flat, 0.0, np.clip(_vec(unrealized_pnl, "unrealized_pnl") / PNL_SCALE, -PNL_CLIP, PNL_CLIP))
    hold = np.where(flat, 0.0, np.minimum(np.maximum(_vec(holding_bars, "holding_bars"), 0.0), HOLD_CAP) / HOLD_SCALE)
    pdd = np.clip(_vec(portfolio_drawdown, "portfolio_drawdown") / PDD_SCALE, 0.0, PDD_CLIP)
    gross = np.clip(_vec(gross_exposure, "gross_exposure"), 0.0, GROSS_CLIP)
    size = np.where(flat, 0.0, np.clip(_vec(position_size, "position_size") / SIZE_SCALE, 0.0, SIZE_CLIP))
    loss = np.clip(_vec(recent_loss, "recent_loss") / LOSS_SCALE, 0.0, LOSS_CLIP)
    cols = [pnl, hold, pdd, gross, size, loss]
    return np.concatenate([onehot, profile_array(profile, n)] + [c[:, None] for c in cols], axis=1).astype(np.float32)


# ----------------------------------------------------------------------------
# Indicators (manual implementations, causal)
# ----------------------------------------------------------------------------
def _pct(s: pd.Series, n: int) -> pd.Series:
    return s / s.shift(n) - 1.0


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    rs = _wilder(gain, n) / _wilder(loss, n)
    return 100.0 - 100.0 / (1.0 + rs)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def adx_di(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    """Wilder ADX with +DI / -DI."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    atr = _wilder(true_range(high, low, close), n)
    plus_di = 100.0 * _wilder(plus_dm, n) / atr
    minus_di = 100.0 * _wilder(minus_dm, n) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return _wilder(dx, n), plus_di, minus_di


# ----------------------------------------------------------------------------
# Risk features (OHLCV + benchmark) and the documented adapter
# ----------------------------------------------------------------------------
def _pctile_rank(s: pd.Series, n: int) -> pd.Series:
    """Percentile rank (0..1) of the latest value within its trailing n values (including itself). Causal."""
    return s.rolling(n, min_periods=n).apply(lambda x: float((x <= x[-1]).mean()), raw=True)


def _rolling_maxdd(c: pd.Series, n: int) -> pd.Series:
    """Worst peak-to-trough decline (<= 0) inside the trailing n bars. Causal."""
    def f(x):
        return float((x / np.maximum.accumulate(x) - 1.0).min())
    return c.rolling(n, min_periods=n).apply(f, raw=True)


def risk_features(ohlcv: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """The 13 SP-05 risk features (EXTRA_FEATURE_COLUMNS). Causal; z-scores compare bar t with the PREVIOUS 60 bars."""
    df = ohlcv[OHLCV_COLUMNS].astype(float)
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    vz = v.where(v > 0)
    ret = c / c.shift(1) - 1.0
    tr = true_range(h, l, c)
    atr14 = _wilder(tr, 14)
    atr_pct = atr14 / c
    lr = np.log(c).diff()
    rv20 = lr.rolling(20).std() * np.sqrt(252.0)
    out = pd.DataFrame(index=df.index)
    out["atr_pct_pctile_252"] = _pctile_rank(atr_pct, 252)            # volatility percentile (ATR based)
    out["rv20_pctile_252"] = _pctile_rank(rv20, 252)                  # volatility percentile (realised)
    out["dd_from_high_60"] = c / c.rolling(60).max() - 1.0
    out["dd_from_high_252"] = c / c.rolling(252).max() - 1.0
    out["maxdd_60"] = _rolling_maxdd(c, 60)                           # rolling maximum drawdown
    out["gap_abs_atr"] = (o - c.shift(1)).abs() / atr14.shift(1)      # gap size in ATRs (abnormal gap indicator)
    sd60 = ret.shift(1).rolling(60).std().replace(0.0, np.nan)
    out["abnormal_ret_z"] = ret / sd60                                # abnormal movement vs the previous 60 bars
    hl = (h - l) / c
    out["range_z"] = (hl - hl.shift(1).rolling(60).mean()) / hl.shift(1).rolling(60).std().replace(0.0, np.nan)
    out["vol_anomaly_z"] = (vz - vz.shift(1).rolling(60).mean()) / vz.shift(1).rolling(60).std().replace(0.0, np.nan)
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    out["bb_width_20"] = (4.0 * s20) / m20                            # Bollinger Band width
    out["down_vol_20"] = np.sqrt((np.minimum(ret, 0.0) ** 2).rolling(20).mean()) * np.sqrt(252.0)   # downside deviation
    bc = benchmark["Close"].astype(float).reindex(df.index).ffill(limit=3)                 # past-only fill
    out["bench_dd_60"] = bc / bc.rolling(60).max() - 1.0              # market stress: benchmark drawdown
    brv20 = np.log(bc).diff().rolling(20).std() * np.sqrt(252.0)
    out["bench_rv20_pctile_252"] = _pctile_rank(brv20, 252)           # market stress: benchmark volatility percentile
    return out[EXTRA_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


class RiskFeatureAdapter:
    """Documented adapter: OHLCV (+ benchmark) history -> the 13 SP-05 risk features, added to a shared state.

    The shared state (sequence (L, 30), position, profile, unrealized_pnl, holding_bars) is unchanged; the adapter
    only adds ``risk_features`` (L, 13). It never downloads data: the caller supplies OHLCV history of at least
    ``min_history`` + seq_len bars ending at the decision bar (for stock and benchmark, same dates).
    """
    min_history = RISK_MIN_HISTORY
    columns = list(EXTRA_FEATURE_COLUMNS)

    def transform(self, ohlcv: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
        return risk_features(ohlcv, benchmark)

    def window(self, ohlcv: pd.DataFrame, benchmark: pd.DataFrame, seq_len: int) -> np.ndarray:
        if len(ohlcv) < self.min_history + seq_len:
            raise ValueError(f"need at least {self.min_history + seq_len} bars of OHLCV history; got {len(ohlcv)}")
        return self.transform(ohlcv, benchmark).iloc[-seq_len:].to_numpy(dtype=np.float32)

    def adapt_state(self, state: dict, ohlcv: pd.DataFrame, benchmark: pd.DataFrame) -> dict:
        seq_len = np.asarray(state["sequence"]).shape[-2]
        return {**state, "risk_features": self.window(ohlcv, benchmark, seq_len)}


# ----------------------------------------------------------------------------
# Feature builder
# ----------------------------------------------------------------------------
def build_features(ohlcv: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Compute causal features for one symbol.

    Parameters
    ----------
    ohlcv : DataFrame indexed by date, columns Open/High/Low/Close/Volume (adjusted prices).
    benchmark : DataFrame indexed by date with at least a Close column.

    Returns a DataFrame with the same index: the OHLCV columns followed by FEATURE_COLUMNS.
    Row t only depends on rows <= t (verified by a perturbation test in the notebook).
    """
    if benchmark is None or "Close" not in benchmark:
        raise ValueError("a benchmark DataFrame with a Close column is required")
    missing = [c for c in OHLCV_COLUMNS if c not in ohlcv.columns]
    if missing:
        raise ValueError(f"ohlcv is missing columns: {missing}")

    df = ohlcv[OHLCV_COLUMNS].astype(float).copy()
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    f = pd.DataFrame(index=df.index)

    # returns
    f["ret_1"] = _pct(c, 1)
    f["logret_1"] = np.log(c).diff()
    for n in (5, 10, 20, 60):
        f[f"ret_{n}"] = _pct(c, n)

    # momentum oscillators
    f["rsi_14"] = rsi(c, 14)
    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    f["macd_pct"] = macd / c
    f["macd_signal_pct"] = macd_signal / c
    f["macd_hist_pct"] = (macd - macd_signal) / c

    # moving-average relationships
    ema20 = c.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = c.ewm(span=50, adjust=False, min_periods=50).mean()
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    f["close_ema20"] = c / ema20 - 1.0
    f["close_ema50"] = c / ema50 - 1.0
    f["ema20_ema50"] = ema20 / ema50 - 1.0
    f["close_sma200"] = c / sma200 - 1.0
    f["sma50_sma200"] = sma50 / sma200 - 1.0

    # trend strength
    adx, plus_di, minus_di = adx_di(h, l, c, 14)
    f["adx_14"] = adx
    f["di_diff"] = plus_di - minus_di

    # volatility
    f["atr_pct"] = _wilder(true_range(h, l, c), 14) / c
    f["rv_20"] = f["logret_1"].rolling(20).std() * np.sqrt(252.0)
    f["rv_60"] = f["logret_1"].rolling(60).std() * np.sqrt(252.0)

    # volume (zero volume treated as missing for ratios; never fabricated)
    vz = v.where(v > 0)
    f["vol_chg_1"] = np.log(vz / vz.shift(1)).clip(-3.0, 3.0)
    f["vol_ratio_20"] = vz / vz.rolling(20).mean()
    f["vol_z_60"] = (vz - vz.rolling(60).mean()) / vz.rolling(60).std().replace(0.0, np.nan)

    # candle shape
    f["hl_range"] = (h - l) / c
    f["gap_open"] = o / c.shift(1) - 1.0
    f["close_loc"] = (c - l) / (h - l).replace(0.0, np.nan)

    # relative strength vs benchmark (past-only forward fill, at most 3 bars)
    bc = benchmark["Close"].astype(float).reindex(df.index).ffill(limit=3)
    b_ret20 = _pct(bc, 20)
    b_ret60 = _pct(bc, 60)
    f["rel_strength_20"] = f["ret_20"] - b_ret20
    f["rel_strength_60"] = f["ret_60"] - b_ret60
    f["bench_ret_20"] = b_ret20
    f["bench_rv_20"] = np.log(bc).diff().rolling(20).std() * np.sqrt(252.0)

    # ---- SP-05 risk extras (same function the adapter exposes) -------------------------------
    ex = risk_features(df, benchmark)
    for col in EXTRA_FEATURE_COLUMNS:
        f[col] = ex[col]

    f = f[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return pd.concat([df, f], axis=1)


def scale_features(x, median, mean, std, clip) -> np.ndarray:
    """Single implementation of the preprocessing used in training AND inference.

    Non-finite values -> training-set median, then z-score, then clip to +/- clip.
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.where(np.isfinite(x), x, np.asarray(median, dtype=np.float32))
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.clip(x, -float(clip), float(clip)).astype(np.float32)


def build_feature_schema(seq_len: int) -> dict:
    """Machine-readable description of the exact model input layout."""
    return {
        "schema_version": "1.0.0",
        "policy_id": "SP-05",
        "sequence_input": {
            "name": "sequence",
            "shape": ["batch", seq_len, N_FEATURES],
            "dtype": "float32",
            "order": "oldest bar first, newest bar last (index -1 is the decision bar)",
            "features": [{"index": i, "name": n} for i, n in enumerate(FEATURE_COLUMNS)],
            "scaling": "NaN -> train median, then (x - mean) / std, then clip to +/- clip (see preprocessing.pkl)",
            "note": "Callers pass RAW (unscaled) features; the policy scales internally.",
        },
        "static_input": {
            "name": "static",
            "shape": ["batch", N_STATIC],
            "extras_note": "indices 0-7 identical to SP-02/03/04; indices 8-11 (portfolio drawdown, gross exposure, position size, recent loss) are SP-05 additions; every extra defaults to 0 when absent",
            "dtype": "float32",
            "features": [{"index": i, "name": n} for i, n in enumerate(STATIC_FEATURE_NAMES)],
            "position_encoding": {"one_hot_order": list(POSITION_NAMES), "position_values": list(POSITION_ORDER)},
            "profile_ranges": {k: [0.0, 1.0] for k in PROFILE_FEATURES},
            "default_profile": dict(zip(PROFILE_FEATURES, DEFAULT_PROFILE)),
        },
        "output": {
            "action_probs": ["batch", N_ACTIONS],
            "action_space": {str(i): n for i, n in enumerate(ACTION_NAMES)},
            "confidence": "temperature-calibrated probability of the chosen action after masking invalid actions",
        },
        "valid_actions": {
            "flat": ["HOLD", "BUY", "SELL (only if allow_short)"],
            "long": ["HOLD", "EXIT"],
            "short": ["HOLD", "EXIT"],
        },
        "execution_convention": "decision at close of bar t; fill at open of bar t+1",
        "compatibility": {
            "family_rule": "every SP-0x policy consumes the shared base features first, then its own extras; extras DIFFER per policy, so select columns by name",
            "shared_sequence_features": list(BASE_FEATURE_COLUMNS),
            "sp05_extra_sequence_features": list(EXTRA_FEATURE_COLUMNS),
            "shared_static_features": list(SHARED_STATIC_NAMES),
            "sp05_extra_static_features": list(STATIC_FEATURE_NAMES[N_SHARED_STATIC:]),
            "state_dict_keys": ["sequence", "position", "profile", "unrealized_pnl (optional)", "holding_bars (optional)", "risk_features (SP-05; from RiskFeatureAdapter)", "portfolio_drawdown (optional)", "gross_exposure (optional)", "position_size (optional)", "recent_loss (optional)"],
            "adapter": {"class": "RiskFeatureAdapter", "input": "stock and benchmark OHLCV history ending at the decision bar (>= seq_len + 290 bars)", "output": "risk_features (L, 13)", "alternative": "pass sequence already extended to 43 columns"},
            "authority": "SP-05 only PROPOSES actions; sp05_risk.RiskController has final authority over execution",
        },
    }