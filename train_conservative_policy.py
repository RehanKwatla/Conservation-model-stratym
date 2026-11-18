"""SP-05 Conservative Risk-Aware Policy — Training, Evaluation & Artifact Pipeline.

Preserves the exact existing architecture from 05_conservative_policy.ipynb:
  - sp05_features.py (43 causal sequence features + 12 static inputs)
  - sp05_model.py    (ConservativePolicyNet: 2-layer GRU + static MLP + action head)
  - sp05_risk.py     (RiskController deterministic safety gate)

Trains on the local Indian market historical OHLCV dataset
(indian-market-historical-ohlcv.csv) with a strict chronological
72% train / 8% validation / 20% untouched test split (80:20 dev:test),
random seed 42, periodic & best checkpointing, temperature scaling on
validation only, and single final evaluation on the untouched test split.
"""
from __future__ import annotations

import json
import os
import pickle
import random
import shutil
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from sp05_features import (
    ACTION_NAMES,
    BASE_FEATURE_COLUMNS,
    BUY,
    DEFAULT_PROFILE,
    EXIT,
    EXTRA_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    HOLD,
    N_ACTIONS,
    N_BASE_FEATURES,
    N_FEATURES,
    N_SHARED_STATIC,
    N_STATIC,
    OHLCV_COLUMNS,
    PROFILE_FEATURES,
    SELL,
    WARMUP_BARS,
    build_features,
    encode_static,
    scale_features,
    valid_action_mask,
)
from sp05_model import ConservativePolicy, ConservativePolicyNet, mask_logits
from sp05_risk import REASON_NAMES, RiskController, RiskLimits

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# CONFIGURATION (preserved from 05_conservative_policy.ipynb, adapted to local dataset)
# ============================================================================
CONFIG = {
    "seed": 42,
    "dataset_csv": "indian-market-historical-ohlcv.csv",
    "benchmark": "NIFTY 50",
    "symbols": [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
        "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK",
        "TITAN", "TATASTEEL", "SUNPHARMA", "WIPRO",
    ],
    "min_history_rows": 800,
    "split_ratios": {"train": 0.72, "val": 0.08, "test": 0.20},
    "seq_len": 30,
    "strict_window_isolation": True,
    "allow_short": True,
    "objective": {
        "w_return": 1.0,
        "w_drawdown": 1.0,
        "w_turnover": 0.3,
        "entry_threshold": 0.3,
        "exit_threshold": -0.25,
        "short_penalty": 0.2,
    },
    "soft_limits": {
        "portfolio_drawdown": 0.05,
        "gross_exposure": 0.50,
        "recent_loss": 0.04,
        "vol_exit_percentile": 0.90,
        "abnormal_ret_z": 3.0,
        "gap_atr": 2.5,
        "market_stress_dd": 0.10,
    },
    "hold_sample_max": 60,
    "eval_horizon": 10,
    "initial_capital": 100000.0,
    "dd_lookback_bars": 120,
    "position_fraction": 0.10,
    "max_position_size": 0.20,
    "max_gross_exposure": 1.0,
    "min_trade_fraction": 0.005,
    "fee_bps": 5.0,
    "slippage_bps": 5.0,
    "short_borrow_bps_annual": 50.0,
    "risk_free_annual": 0.0,
    "confidence_sizing": True,
    "risk_limits": {
        "max_position_fraction": 0.10,
        "max_gross_exposure": 0.60,
        "position_stop_loss": 0.05,
        "max_holding_bars": 120,
        "halt_drawdown": 0.08,
        "flatten_drawdown": 0.15,
        "recent_loss_halt": 0.06,
        "max_vol_percentile": 0.90,
        "max_abs_gap_atr": 3.0,
        "max_abnormal_ret_z": 4.0,
        "market_stress_dd": 0.12,
        "min_entry_confidence": 0.50,
    },
    "model": {
        "encoder": "gru",
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.25,
        "static_hidden": 16,
        "head_hidden": 64,
    },
    "train": {
        "epochs": 40,
        "batch_size": 512,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 7,
        "grad_clip": 1.0,
        "input_noise_std": 0.02,
        "max_class_weight": 20.0,
        "scaler_clip": 5.0,
        "checkpoint_every": 5,
    },
    "artifact_dir": "./sp05_conservative_artifacts",
    "checkpoint_dir": "./checkpoints",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_py(o):
    if isinstance(o, pd.DataFrame):
        return to_py(o.reset_index().to_dict(orient="records"))
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_py(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


# ============================================================================
# STEP 1: LOAD & CLEAN LOCAL CSV DATASET
# ============================================================================
def clean_ohlcv(df: pd.DataFrame, sym: str):
    """Apply split/dividend adjustment via adj_close/close and clean OHLCV bars."""
    stats = {"symbol": sym, "raw_rows": int(len(df))}
    df = df.copy()

    # Rename lowercase columns to standard OHLCV
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "adj_close": "Adj_Close",
    }
    df = df.rename(columns=rename_map)

    for col in OHLCV_COLUMNS + ["Adj_Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    idx = pd.to_datetime(df["date"])
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = pd.DatetimeIndex(idx).normalize()
    df.index.name = "Date"

    stats["duplicate_dates_removed"] = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="last")].sort_index()

    price = ["Open", "High", "Low", "Close"]
    bad = (
        df[price].isna().any(axis=1)
        | (df[price] <= 0).any(axis=1)
        | (df["High"] < df["Low"])
    )
    stats["bad_price_rows_dropped"] = int(bad.sum())
    df = df.loc[~bad].copy()

    # Apply split/dividend adjustment consistent with yfinance auto_adjust=True
    if "Adj_Close" in df.columns:
        valid_adj = df["Adj_Close"].notna() & (df["Adj_Close"] > 0) & (df["Close"] > 0)
        ratio = np.where(valid_adj, df["Adj_Close"] / df["Close"], 1.0)
        for col in price:
            df[col] = df[col] * ratio

    df = df[OHLCV_COLUMNS].copy()
    df.loc[df["Volume"] < 0, "Volume"] = np.nan
    stats["missing_volume_rows"] = int(df["Volume"].isna().sum())
    stats["zero_volume_rows"] = int((df["Volume"] == 0).sum())
    stats["extreme_moves_gt_40pct"] = int((df["Close"].pct_change().abs() > 0.40).sum())
    stats["rows"] = int(len(df))
    stats["first_date"] = str(df.index.min().date()) if len(df) else None
    stats["last_date"] = str(df.index.max().date()) if len(df) else None
    return df, stats


def load_local_dataset(csv_path: Path, symbols: list[str], benchmark: str):
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    needed = set(symbols + [benchmark])
    print(f"Loading symbols ({len(symbols)} stocks + benchmark '{benchmark}') from {csv_path.name}...")
    chunks = pd.read_csv(
        csv_path,
        usecols=["date", "open", "high", "low", "close", "adj_close", "volume", "symbol"],
        chunksize=500_000,
    )
    matched = []
    total_csv_rows = 0
    for chunk in chunks:
        total_csv_rows += len(chunk)
        sub = chunk[chunk["symbol"].isin(needed)]
        if len(sub) > 0:
            matched.append(sub)

    full = pd.concat(matched, ignore_index=True)
    print(f"Scanned {total_csv_rows:,} total CSV rows; extracted {len(full):,} rows for target universe.")

    b_raw = full[full["symbol"] == benchmark]
    if len(b_raw) == 0:
        raise RuntimeError(f"Benchmark '{benchmark}' not found in {csv_path}")
    bench_df, b_stats = clean_ohlcv(b_raw, benchmark)
    b_stats["status"] = "ok"
    data_log = [b_stats]

    bench_start, bench_end = bench_df.index.min(), bench_df.index.max()
    raw_data = {}
    for sym in symbols:
        s_raw = full[full["symbol"] == sym]
        if len(s_raw) == 0:
            print(f"WARNING: {sym} not found in CSV; skipping.")
            continue
        s_df, s_stats = clean_ohlcv(s_raw, sym)
        # Keep history slightly before benchmark start for stock indicator warmup, up to benchmark end
        s_df = s_df.loc[:bench_end]
        if len(s_df) < CONFIG["min_history_rows"]:
            print(f"WARNING: {sym} has only {len(s_df)} rows; skipping.")
            continue
        s_stats["status"] = "ok"
        raw_data[sym] = s_df
        data_log.append(s_stats)

    return bench_df, raw_data, pd.DataFrame(data_log).set_index("symbol"), total_csv_rows


# ============================================================================
# STEP 2: CAUSAL FEATURES & CAUSALITY VERIFICATION
# ============================================================================
def verify_causality(raw_data: dict, bench_df: pd.DataFrame) -> None:
    sym = list(raw_data)[0]
    ohlcv = raw_data[sym]
    n = int(len(ohlcv) * 0.7)
    base = build_features(ohlcv, bench_df)[FEATURE_COLUMNS].iloc[:n].to_numpy()
    rng = np.random.default_rng(0)
    pert = ohlcv.copy()
    pert.iloc[n:, :] = pert.iloc[n:, :].to_numpy() * rng.uniform(0.3, 3.0, size=pert.iloc[n:].shape)
    pert_bench = bench_df.copy()
    cutoff_date = ohlcv.index[n]
    pert_bench.loc[pert_bench.index >= cutoff_date, :] *= 2.5
    after = build_features(pert, pert_bench)[FEATURE_COLUMNS].iloc[:n].to_numpy()
    same = np.allclose(base, after, equal_nan=True, atol=1e-12)
    assert same, "CAUSALITY TEST FAILED: future perturbation changed past features!"
    print(f"Causality perturbation test PASSED on {sym} (zero future leakage verified).")


def build_panel(raw_data: dict, bench_df: pd.DataFrame) -> pd.DataFrame:
    H = CONFIG["eval_horizon"]
    bench_start = bench_df.index.min()
    frames = []
    for sym, ohlcv in raw_data.items():
        feat = build_features(ohlcv, bench_df)
        # Restrict to dates where benchmark exists (after benchmark warmup)
        feat = feat.loc[feat.index >= bench_start].copy()
        # Drop initial WARMUP_BARS rows per symbol so rolling 252-bar features are populated
        feat = feat.iloc[WARMUP_BARS:].copy()
        feat["fwd_ret_h"] = feat["Close"].shift(-H) / feat["Close"] - 1.0  # evaluation only
        feat["symbol"] = sym
        feat["date"] = feat.index
        feat["row_in_sym"] = np.arange(len(feat), dtype=np.int64)
        frames.append(feat.reset_index(drop=True))
    panel = pd.concat(frames, ignore_index=True)
    print(f"Built feature panel: {len(panel):,} rows across {len(raw_data)} symbols "
          f"({panel['date'].min().date()} -> {panel['date'].max().date()}).")
    return panel


# ============================================================================
# STEP 3: CONSERVATIVE OBJECTIVE LABELS & STRICT 72 / 8 / 20 SPLIT
# ============================================================================
def cons_params(profile: np.ndarray) -> dict:
    OBJ = CONFIG["objective"]
    risk, pat, agg = profile[:, 0], profile[:, 1], profile[:, 2]
    return {
        "w_ret": OBJ["w_return"],
        "w_dd": OBJ["w_drawdown"] * (1.5 - risk),
        "w_turn": OBJ["w_turnover"] * (0.5 + pat),
        "thr": OBJ["entry_threshold"] + 0.5 * (1.0 - agg),
        "soft_stop": 0.015 + 0.03 * risk,
        "vol_cap": 0.6 + 0.2 * risk,
    }


def decision_scores(feat: dict):
    with np.errstate(invalid="ignore", divide="ignore"):
        sharpe60 = np.clip(feat["ret_60"] / (feat["rv_60"] * np.sqrt(60.0 / 252.0)), -3.0, 3.0)
        struct = 0.5 * np.sign(feat["sma50_sma200"]) + 0.5 * np.sign(feat["close_ema50"])
        edge = 0.6 * sharpe60 + 0.6 * struct
        vol_pct = 0.5 * (feat["atr_pct_pctile_252"] + feat["rv20_pctile_252"])
        risk = vol_pct + 0.5 * np.clip(-feat["dd_from_high_60"] / 0.10, 0.0, 2.0)
    return edge, risk, vol_pct


def conservative_rule_labels(feat, pos, profile, hold, pnl, port, allow_short=True):
    OBJ, SOFT = CONFIG["objective"], CONFIG["soft_limits"]
    p = cons_params(profile)
    edge, risk, vol_pct = decision_scores(feat)
    pdd, gross, rloss = (np.asarray(port[k], dtype=float) for k in ("pdd", "gross", "rloss"))
    with np.errstate(invalid="ignore"):
        prohibited = (
            (vol_pct >= p["vol_cap"])
            | (np.abs(feat["abnormal_ret_z"]) >= SOFT["abnormal_ret_z"])
            | (feat["gap_abs_atr"] >= SOFT["gap_atr"])
            | (feat["bench_dd_60"] <= -SOFT["market_stress_dd"])
        )
        portfolio_block = (
            (pdd >= SOFT["portfolio_drawdown"])
            | (gross >= SOFT["gross_exposure"])
            | (rloss >= SOFT["recent_loss"])
        )
        u_long = p["w_ret"] * edge - p["w_dd"] * risk - p["w_turn"]
        u_short = -p["w_ret"] * edge - p["w_dd"] * risk - p["w_turn"] - OBJ["short_penalty"]
        long_entry = (u_long >= p["thr"]) & ~prohibited & ~portfolio_block
        short_entry = (u_short >= p["thr"]) & ~prohibited & ~portfolio_block
        h_long = p["w_ret"] * edge - p["w_dd"] * risk
        h_short = -p["w_ret"] * edge - p["w_dd"] * risk
        vol_spike = vol_pct >= SOFT["vol_exit_percentile"]
        derisk = (pdd >= SOFT["portfolio_drawdown"]) | ((rloss >= SOFT["recent_loss"]) & (pnl < 0))
        long_exit = (
            (pnl <= -p["soft_stop"])
            | (h_long <= OBJ["exit_threshold"])
            | vol_spike
            | (feat["abnormal_ret_z"] <= -SOFT["abnormal_ret_z"])
            | derisk
        )
        short_exit = (
            (pnl <= -p["soft_stop"])
            | (h_short <= OBJ["exit_threshold"])
            | vol_spike
            | (feat["abnormal_ret_z"] >= SOFT["abnormal_ret_z"])
            | derisk
        )
    y = np.zeros(len(pos), dtype=np.int64)
    flat = pos == 0
    y[flat & long_entry] = BUY
    if allow_short:
        y[flat & ~long_entry & short_entry] = SELL
    y[(pos == 1) & long_exit] = EXIT
    y[(pos == -1) & short_exit] = EXIT
    return y


# ============================================================================
# STEP 4: EVALUATION & CALIBRATION HELPERS
# ============================================================================
def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = Fnn.cross_entropy(logits / torch.exp(log_t), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).clamp(0.05, 10.0).item())


def ece_score(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
            rows.append({
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": int(m.sum()),
                "confidence": float(conf[m].mean()),
                "accuracy": float(correct[m].mean()),
            })
    return float(ece), rows


def classification_summary(logits: torch.Tensor, y, temperature: float) -> dict:
    y = np.asarray(y)
    probs = torch.softmax(logits / temperature, dim=1).numpy()
    pred = probs.argmax(1)
    conf = probs.max(1)
    labels = list(range(N_ACTIONS))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y, pred, labels=labels, average="macro", zero_division=0)
    p_wt, r_wt, f_wt, _ = precision_recall_fscore_support(y, pred, labels=labels, average="weighted", zero_division=0)
    ece, bins = ece_score(conf, (pred == y).astype(float))
    return {
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(p_mac),
        "macro_recall": float(r_mac),
        "macro_f1": float(f_mac),
        "weighted_precision": float(p_wt),
        "weighted_recall": float(r_wt),
        "weighted_f1": float(f_wt),
        "majority_class_baseline_accuracy": float(np.bincount(y, minlength=N_ACTIONS).max() / len(y)),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean()),
        "ece": ece,
        "reliability_bins": bins,
        "temperature": float(temperature),
        "per_class": {
            ACTION_NAMES[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i in labels
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "classification_report_text": classification_report(
            y, pred, labels=labels, target_names=ACTION_NAMES, digits=4, zero_division=0
        ),
    }


# ============================================================================
# STEP 5: BACKTEST ENGINE (WITH & WITHOUT DETERMINISTIC RISK GATE)
# ============================================================================
def run_portfolio_backtest(
    panel: pd.DataFrame,
    raw_mat: np.ndarray,
    rows: np.ndarray,
    bench_df: pd.DataFrame,
    policy: ConservativePolicy,
    controller: RiskController,
    use_gate: bool = True,
    profile=(0.5, 0.5, 0.5),
):
    L = CONFIG["seq_len"]
    offsets = np.arange(-(L - 1), 1)
    fee = CONFIG["fee_bps"] / 1e4
    slip = CONFIG["slippage_bps"] / 1e4
    borrow_daily = CONFIG["short_borrow_bps_annual"] / 1e4 / 252.0
    cap0 = float(CONFIG["initial_capital"])
    pos_frac = CONFIG["position_fraction"]
    max_gross = CONFIG["max_gross_exposure"]
    dd_lb = CONFIG["dd_lookback_bars"]

    sub = panel.loc[rows, ["date", "symbol", "Open", "Close"]].copy()
    sub["row_id"] = rows
    dates = np.sort(sub["date"].unique())
    syms = sorted(sub["symbol"].unique())
    sym_to_i = {s: i for i, s in enumerate(syms)}
    S_cnt = len(syms)

    # Pre-index by date
    by_date = {d: grp for d, grp in sub.groupby("date")}

    # Benchmark returns aligned to dates
    b_close = bench_df["Close"].reindex(dates).ffill()
    bench_ret = b_close.pct_change().fillna(0.0).to_numpy()

    cash = cap0
    shares = np.zeros(S_cnt, dtype=float)
    entry_px = np.zeros(S_cnt, dtype=float)
    hold_bars = np.zeros(S_cnt, dtype=int)
    last_close = np.zeros(S_cnt, dtype=float)

    eq_hist = [cap0]
    ret_hist = []
    ew_ret_hist = []
    exp_hist = []
    dd_hist = []
    gate_overrides = 0
    reason_counts = {r: 0 for r in REASON_NAMES}
    n_trades = 0

    for t in range(len(dates) - 1):
        d_cur = dates[t]
        d_nxt = dates[t + 1]
        g_cur = by_date[d_cur]
        g_nxt = by_date[d_nxt]

        # Map next-day opens and closes
        nxt_open = {}
        nxt_close = {}
        for _, r_n in g_nxt.iterrows():
            si = sym_to_i[r_n["symbol"]]
            nxt_open[si] = float(r_n["Open"])
            nxt_close[si] = float(r_n["Close"])

        # Update current closes
        cur_rows = []
        cur_sis = []
        for _, r_c in g_cur.iterrows():
            si = sym_to_i[r_c["symbol"]]
            last_close[si] = float(r_c["Close"])
            if si in nxt_open:
                cur_rows.append(int(r_c["row_id"]))
                cur_sis.append(si)

        # Equal weight daily return over available symbols
        ew_r = []
        for si in cur_sis:
            if last_close[si] > 0 and si in nxt_close:
                ew_r.append(nxt_close[si] / last_close[si] - 1.0)
        ew_ret_hist.append(float(np.mean(ew_r)) if ew_r else 0.0)

        # Mark-to-market equity at current close
        equity_cur = cash + float(np.sum(shares * last_close))
        if equity_cur <= 0:
            equity_cur = 1e-6

        recent_eq = eq_hist[-dd_lb:]
        trail_peak = max(recent_eq)
        pdd = max(0.0, 1.0 - equity_cur / trail_peak)
        eq_20 = eq_hist[-20] if len(eq_hist) >= 20 else eq_hist[0]
        rloss = max(0.0, 1.0 - equity_cur / eq_20)
        gross_exp = float(np.sum(np.abs(shares * last_close))) / equity_cur

        if len(cur_rows) > 0:
            r_arr = np.asarray(cur_rows, dtype=np.int64)
            si_arr = np.asarray(cur_sis, dtype=np.int64)
            win = raw_mat[r_arr[:, None] + offsets]  # (B, L, 43)

            pos_vec = np.sign(shares[si_arr]).astype(int)
            pnl_vec = np.where(
                pos_vec != 0,
                pos_vec * (last_close[si_arr] / np.where(entry_px[si_arr] > 0, entry_px[si_arr], 1.0) - 1.0),
                0.0,
            )
            hold_vec = np.where(pos_vec != 0, hold_bars[si_arr], 0)
            size_vec = np.abs(shares[si_arr] * last_close[si_arr]) / equity_cur

            state = {
                "sequence": win,
                "position": pos_vec,
                "profile": profile,
                "unrealized_pnl": pnl_vec,
                "holding_bars": hold_vec,
                "portfolio_drawdown": pdd,
                "gross_exposure": gross_exp,
                "position_size": size_vec,
                "recent_loss": rloss,
            }
            prop = policy.predict(state)
            proposed = prop["action"]
            conf = prop["confidence"]

            if use_gate:
                last_bar = win[:, -1, :]
                feat_dict = {name: last_bar[:, i] for i, name in enumerate(FEATURE_COLUMNS)}
                port_dict = {
                    "portfolio_drawdown": pdd,
                    "gross_exposure": gross_exp,
                    "recent_loss": rloss,
                    "entry_fraction": pos_frac,
                }
                actions, reasons = controller.review_batch(
                    proposed, conf, pos_vec, pnl_vec, hold_vec, feat_dict, port_dict
                )
                overridden = actions != proposed
                gate_overrides += int(overridden.sum())
                for ri, rname in enumerate(REASON_NAMES):
                    reason_counts[rname] += int(reasons[:, ri].sum())
            else:
                actions = proposed

            # Execute at next bar open (t+1)
            running_gross = float(np.sum(np.abs(shares * last_close))) / equity_cur
            # First process exits
            for k, si in enumerate(si_arr):
                act = int(actions[k])
                px_fill = nxt_open[si]
                if act == EXIT and shares[si] != 0:
                    qty = shares[si]
                    fill = px_fill * (1.0 - slip) if qty > 0 else px_fill * (1.0 + slip)
                    notional = abs(qty * fill)
                    cash += qty * fill - notional * fee
                    shares[si] = 0.0
                    entry_px[si] = 0.0
                    hold_bars[si] = 0
                    n_trades += 1
            running_gross = float(np.sum(np.abs(shares * last_close))) / equity_cur

            # Next process entries
            for k in np.argsort(-conf):
                si = int(si_arr[k])
                act = int(actions[k])
                if shares[si] != 0 or act not in (BUY, SELL):
                    continue
                px_fill = nxt_open[si]
                if px_fill <= 0:
                    continue
                target_f = pos_frac * (float(conf[k]) if CONFIG["confidence_sizing"] else 1.0)
                target_f = min(target_f, CONFIG["max_position_size"])
                if running_gross + target_f > max_gross:
                    continue
                notional = target_f * equity_cur
                if notional < CONFIG["min_trade_fraction"] * equity_cur:
                    continue
                if act == BUY:
                    fill = px_fill * (1.0 + slip)
                    qty = notional / fill
                    cash -= qty * fill + notional * fee
                    shares[si] = qty
                else:
                    fill = px_fill * (1.0 - slip)
                    qty = -notional / fill
                    cash -= qty * fill + notional * fee
                    shares[si] = qty
                entry_px[si] = fill
                hold_bars[si] = 1
                running_gross += target_f
                n_trades += 1

        # Borrow cost on short positions and advance hold bars
        short_notional = float(np.sum(np.abs(np.minimum(shares, 0.0) * last_close)))
        cash -= short_notional * borrow_daily
        for si in range(S_cnt):
            if shares[si] != 0:
                hold_bars[si] += 1
            if si in nxt_close:
                last_close[si] = nxt_close[si]

        equity_nxt = cash + float(np.sum(shares * last_close))
        r_day = equity_nxt / equity_cur - 1.0
        eq_hist.append(equity_nxt)
        ret_hist.append(r_day)
        exp_hist.append(float(np.sum(np.abs(shares * last_close))) / max(equity_nxt, 1e-6))
        peak_nxt = max(eq_hist[-dd_lb:])
        dd_hist.append(max(0.0, 1.0 - equity_nxt / peak_nxt))

    ret_arr = np.asarray(ret_hist, dtype=float)
    eq_arr = np.asarray(eq_hist[1:], dtype=float) / cap0
    ann_ret = float((eq_arr[-1] ** (252.0 / max(len(ret_arr), 1))) - 1.0) if len(eq_arr) else 0.0
    ann_vol = float(ret_arr.std() * np.sqrt(252.0)) if len(ret_arr) > 1 else 0.0
    sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-9 else 0.0
    peak_all = np.maximum.accumulate(np.concatenate([[1.0], eq_arr]))[1:]
    max_dd = float((eq_arr / peak_all - 1.0).min()) if len(eq_arr) else 0.0
    calmar = float(ann_ret / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0

    return {
        "dates": dates[1:],
        "ret": ret_arr,
        "equity": eq_arr,
        "bench_ret": bench_ret[1:],
        "ew_ret": np.asarray(ew_ret_hist, dtype=float),
        "exposure": np.asarray(exp_hist, dtype=float),
        "portfolio_drawdown": np.asarray(dd_hist, dtype=float),
        "metrics": {
            "total_return": float(eq_arr[-1] - 1.0) if len(eq_arr) else 0.0,
            "cagr": ann_ret,
            "annual_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "avg_gross_exposure": float(np.mean(exp_hist)) if exp_hist else 0.0,
            "n_trades": int(n_trades),
            "gate_overrides": int(gate_overrides),
        },
        "gate_reason_counts": reason_counts,
    }


def bh_metrics(ret_arr: np.ndarray) -> dict:
    eq = np.cumprod(1.0 + ret_arr)
    ann_ret = float((eq[-1] ** (252.0 / max(len(ret_arr), 1))) - 1.0) if len(eq) else 0.0
    ann_vol = float(ret_arr.std() * np.sqrt(252.0)) if len(ret_arr) > 1 else 0.0
    sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-9 else 0.0
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    max_dd = float((eq / peak - 1.0).min()) if len(eq) else 0.0
    return {
        "total_return": float(eq[-1] - 1.0) if len(eq) else 0.0,
        "cagr": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    t_start = time.time()
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print("SP-05 CONSERVATIVE RISK-AWARE POLICY — TRAINING & EVALUATION PIPELINE")
    print("=" * 88)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"Random seed: {CONFIG['seed']}")

    art_dir = Path(CONFIG["artifact_dir"])
    fig_dir = art_dir / "figures"
    ckpt_dir = Path(CONFIG["checkpoint_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    csv_path = Path(CONFIG["dataset_csv"])
    bench_df, raw_data, data_report, total_csv_rows = load_local_dataset(
        csv_path, CONFIG["symbols"], CONFIG["benchmark"]
    )
    print(data_report[["rows", "first_date", "last_date", "bad_price_rows_dropped", "duplicate_dates_removed"]].to_string())

    # 2. Causality test & feature panel
    verify_causality(raw_data, bench_df)
    panel = build_panel(raw_data, bench_df)

    # 3. Strict chronological 72% / 8% / 20% split (80% dev : 20% untouched test)
    L = CONFIG["seq_len"]
    win_start = panel.groupby("symbol")["date"].shift(L - 1)

    all_dates = np.sort(panel["date"].unique())
    n_dates = len(all_dates)
    idx_train_end = int(round(CONFIG["split_ratios"]["train"] * n_dates)) - 1
    idx_val_end = int(round((CONFIG["split_ratios"]["train"] + CONFIG["split_ratios"]["val"]) * n_dates)) - 1

    d_train_end = pd.Timestamp(all_dates[idx_train_end])
    d_val_end = pd.Timestamp(all_dates[idx_val_end])
    split_bounds = {
        "train": (None, d_train_end),
        "val": (d_train_end + pd.Timedelta(days=1), d_val_end),
        "test": (d_val_end + pd.Timedelta(days=1), None),
    }
    print(f"\nChronological Split Boundaries (Total panel dates: {n_dates:,}):")
    print(f"  TRAIN: <= {d_train_end.date()} ({idx_train_end + 1:,} dates)")
    print(f"  VAL  : {(d_train_end + pd.Timedelta(days=1)).date()} -> {d_val_end.date()} ({idx_val_end - idx_train_end:,} dates)")
    print(f"  TEST : >= {(d_val_end + pd.Timedelta(days=1)).date()} ({n_dates - idx_val_end - 1:,} dates) [UNTOUCHED]")

    def eligible_rows(split: str) -> np.ndarray:
        start, end = split_bounds[split]
        ok = win_start.notna().to_numpy().copy()
        if start is not None:
            ok &= (panel["date"] >= start).to_numpy()
            if CONFIG["strict_window_isolation"]:
                ok &= (win_start >= start).to_numpy()
        if end is not None:
            ok &= (panel["date"] <= end).to_numpy()
        return np.where(ok)[0]

    # 4. Fit preprocessing scaler EXCLUSIVELY on TRAIN rows (zero data leakage)
    raw_dict = {c: panel[c].to_numpy(dtype=float) for c in FEATURE_COLUMNS}
    raw_mat = panel[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    open_arr = panel["Open"].to_numpy(dtype=float)
    close_arr = panel["Close"].to_numpy(dtype=float)
    row_in_sym = panel["row_in_sym"].to_numpy()

    fit_mask = (panel["date"] <= d_train_end).to_numpy()
    med = np.nanmedian(raw_mat[fit_mask], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    imp = np.where(np.isfinite(raw_mat[fit_mask]), raw_mat[fit_mask], med)
    mean = imp.mean(axis=0)
    std = imp.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    preprocessing = {
        "feature_columns": list(FEATURE_COLUMNS),
        "median": med.astype(np.float32),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "clip": float(CONFIG["train"]["scaler_clip"]),
        "fit_rows_end_date": str(d_train_end.date()),
        "n_fit_rows": int(fit_mask.sum()),
    }
    x_scaled = scale_features(
        raw_mat, preprocessing["median"], preprocessing["mean"], preprocessing["std"], preprocessing["clip"]
    )
    print(f"Scaler fit EXCLUSIVELY on {preprocessing['n_fit_rows']:,} TRAIN rows (<= {d_train_end.date()}).")

    # 5. Build position-conditioned samples for train / val / test
    rng = np.random.default_rng(CONFIG["seed"])

    def build_rule_samples(split: str) -> dict:
        rows = eligible_rows(split)
        r = np.repeat(rows, 3)
        n = len(r)
        pos = np.tile(np.array([0, 1, -1]), len(rows))
        prof = rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32)
        cap = np.minimum(CONFIG["hold_sample_max"], row_in_sym[r] + 1)
        hold = np.where(pos == 0, 0, 1 + np.floor(rng.random(n) * cap).astype(int))
        entry_px = open_arr[r - np.maximum(hold, 1) + 1]
        pnl = np.where(pos == 0, 0.0, pos * (close_arr[r] / entry_px - 1.0))
        size = np.where(pos == 0, 0.0, rng.uniform(0.02, 0.15, n))
        port = {
            "pdd": np.where(rng.random(n) < 0.55, rng.random(n) * 0.02, rng.random(n) * 0.15),
            "gross": np.maximum(rng.random(n) * 0.9, size),
            "size": size,
            "rloss": np.where(rng.random(n) < 0.6, 0.0, rng.random(n) * 0.08),
        }
        y = conservative_rule_labels(
            {k: v[r] for k, v in raw_dict.items()}, pos, prof, hold, pnl, port, CONFIG["allow_short"]
        )
        valid = valid_action_mask(pos, CONFIG["allow_short"])
        assert valid[np.arange(len(y)), y].all(), "Invalid action label detected!"
        static = encode_static(pos, prof, pnl, hold, port["pdd"], port["gross"], port["size"], port["rloss"])
        return {
            "row": r.astype(np.int64),
            "pos": pos.astype(np.int64),
            "profile": prof,
            "hold": hold,
            "pnl": pnl,
            "port": port,
            "y": y,
            "valid": valid,
            "static": static,
        }

    samples = {s: build_rule_samples(s) for s in ("train", "val", "test")}
    total_samples = sum(len(samples[s]["y"]) for s in ("train", "val", "test"))

    # Chronology verification
    assert panel.loc[samples["train"]["row"], "date"].max() < panel.loc[samples["val"]["row"], "date"].min()
    assert panel.loc[samples["val"]["row"], "date"].max() < panel.loc[samples["test"]["row"], "date"].min()
    print("Chronology assertion PASSED: max(train_date) < min(val_date) and max(val_date) < min(test_date).")

    split_summary = {}
    print("\nExact Dataset Split Summary:")
    for s in ("train", "val", "test"):
        n_s = len(samples[s]["y"])
        pct = 100.0 * n_s / total_samples
        d_min = str(panel.loc[samples[s]["row"], "date"].min().date())
        d_max = str(panel.loc[samples[s]["row"], "date"].max().date())
        dist_counts = np.bincount(samples[s]["y"], minlength=N_ACTIONS).tolist()
        split_summary[s] = {
            "n_samples": int(n_s),
            "percentage": round(pct, 2),
            "start_date": d_min,
            "end_date": d_max,
            "class_counts": dict(zip(ACTION_NAMES, dist_counts)),
        }
        print(f"  {s.upper():5s}: {n_s:>9,} samples ({pct:5.2f}%) | {d_min} -> {d_max} | classes={dict(zip(ACTION_NAMES, dist_counts))}")
    dev_pct = split_summary["train"]["percentage"] + split_summary["val"]["percentage"]
    print(f"  DEV (Train+Val): {split_summary['train']['n_samples'] + split_summary['val']['n_samples']:>9,} ({dev_pct:5.2f}%) | TEST: {split_summary['test']['n_samples']:>9,} ({split_summary['test']['percentage']:5.2f}%)")

    split_config = {
        "seed": CONFIG["seed"],
        "split_method": "chronological_strict_window_isolation",
        "strict_window_isolation": CONFIG["strict_window_isolation"],
        "seq_len": L,
        "total_samples": int(total_samples),
        "development_percentage": round(dev_pct, 2),
        "test_percentage": split_summary["test"]["percentage"],
        "splits": split_summary,
    }
    (art_dir / "split_config.json").write_text(json.dumps(split_config, indent=2))

    # Class weights computed on TRAIN only
    cnt = np.bincount(samples["train"]["y"], minlength=N_ACTIONS).astype(float)
    w = np.where(cnt > 0, cnt.sum() / (N_ACTIONS * np.maximum(cnt, 1.0)), 0.0)
    w = np.minimum(w, CONFIG["train"]["max_class_weight"])
    class_weights = (w * (cnt.sum() / (w * cnt).sum())).astype(np.float32)
    print("Class weights (train only):", {n: round(float(v), 3) for n, v in zip(ACTION_NAMES, class_weights)})

    # 6. Model instantiation
    model_cfg = {"n_features": N_FEATURES, "n_static": N_STATIC, "n_actions": N_ACTIONS, **CONFIG["model"]}
    net = ConservativePolicyNet(**model_cfg).to(device)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\nModel: ConservativePolicyNet ({CONFIG['model']['encoder'].upper()}, trainable params: {n_params:,})")

    # Move tensors to device
    x_all = torch.from_numpy(x_scaled).to(device)
    offsets_t = torch.arange(-(L - 1), 1, device=device)
    offsets_np = np.arange(-(L - 1), 1)

    def to_torch(S):
        return {
            "row": torch.from_numpy(S["row"]).long().to(device),
            "static": torch.from_numpy(S["static"]).to(device),
            "y": torch.from_numpy(S["y"]).long().to(device),
            "valid": torch.from_numpy(S["valid"]).to(device),
        }

    ts = {k: to_torch(v) for k, v in samples.items()}

    def get_batch(T, idx):
        win = T["row"][idx].unsqueeze(1) + offsets_t
        return x_all[win], T["static"][idx], T["valid"][idx], T["y"][idx]

    @torch.no_grad()
    def forward_all(model, T, batch_size=4096):
        model.eval()
        out = []
        for i in range(0, len(T["y"]), batch_size):
            idx = torch.arange(i, min(i + batch_size, len(T["y"])), device=device)
            seq, st, valid, _ = get_batch(T, idx)
            out.append(mask_logits(model(seq, st), valid).cpu())
        return torch.cat(out)

    # 7. Training loop with periodic & best checkpointing
    tc = CONFIG["train"]
    class_w = torch.tensor(class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    y_val_cpu = ts["val"]["y"].cpu()
    gen = torch.Generator().manual_seed(CONFIG["seed"])
    n_train = len(ts["train"]["y"])

    history = []
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    print("\nStarting Training...")
    t_train_start = time.time()
    for epoch in range(1, tc["epochs"] + 1):
        net.train()
        perm = torch.randperm(n_train, generator=gen).to(device)
        total_loss = 0.0
        correct_train = 0
        seen = 0

        for i in range(0, n_train, tc["batch_size"]):
            idx = perm[i : i + tc["batch_size"]]
            seq, st, valid, y = get_batch(ts["train"], idx)
            if tc["input_noise_std"] > 0:
                seq = seq + tc["input_noise_std"] * torch.randn_like(seq)
            logits = mask_logits(net(seq, st), valid)
            loss = Fnn.cross_entropy(logits, y, weight=class_w)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}, batch {i}!")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), tc["grad_clip"])
            optimizer.step()

            bs = len(idx)
            total_loss += loss.item() * bs
            correct_train += int((logits.argmax(dim=1) == y).sum().item())
            seen += bs

        train_loss = total_loss / seen
        train_acc = correct_train / seen

        val_logits = forward_all(net, ts["val"])
        val_loss = Fnn.cross_entropy(val_logits, y_val_cpu, weight=class_w.cpu()).item()
        val_preds = val_logits.argmax(1).numpy()
        val_y_np = y_val_cpu.numpy()
        val_acc = float(accuracy_score(val_y_np, val_preds))
        val_f1 = float(f1_score(val_y_np, val_preds, average="macro", labels=list(range(N_ACTIONS)), zero_division=0))

        scheduler.step(val_loss)
        lr_cur = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "lr": lr_cur,
            "elapsed_s": round(time.time() - t_train_start, 2),
        })

        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": best_val,
                    "val_acc": val_acc,
                    "val_macro_f1": val_f1,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model_cfg,
                },
                ckpt_dir / "best_checkpoint.pt",
            )
        else:
            bad_epochs += 1

        if epoch % tc["checkpoint_every"] == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "model_state_dict": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_dir / f"checkpoint_epoch_{epoch:02d}.pt",
            )

        print(
            f"Epoch {epoch:2d}/{tc['epochs']} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
            f"| train_acc {train_acc:.4f} | val_acc {val_acc:.4f} | val_F1 {val_f1:.4f} "
            f"| lr {lr_cur:.1e}{' *' if improved else ''} [{time.time() - t_train_start:.1f}s]",
            flush=True,
        )

        if bad_epochs >= tc["patience"]:
            print(f"Early stopping triggered at epoch {epoch} (no val improvement for {tc['patience']} epochs).")
            break

    train_duration = time.time() - t_train_start
    print(f"\nTraining complete in {train_duration:.1f}s. Best epoch: {best_epoch} (val_loss={best_val:.4f}).")

    # Restore best checkpoint
    net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    net.eval()

    history_df = pd.DataFrame(history)
    history_df.to_csv(art_dir / "training_history.csv", index=False)

    # 8. Fit temperature calibration on VALIDATION ONLY, then evaluate ONCE on UNTOUCHED TEST
    val_logits = forward_all(net, ts["val"])
    temperature = fit_temperature(val_logits, ts["val"]["y"].cpu())
    print(f"\nFitted temperature scaling (VALIDATION ONLY): T = {temperature:.4f}")

    test_logits = forward_all(net, ts["test"])
    val_summary = classification_summary(val_logits, samples["val"]["y"], temperature)
    test_summary = classification_summary(test_logits, samples["test"]["y"], temperature)
    test_uncal = classification_summary(test_logits, samples["test"]["y"], 1.0)

    for split_name, summ in [("VAL", val_summary), ("TEST (UNTOUCHED)", test_summary)]:
        print(f"\n--- {split_name} METRICS ---")
        print(
            f"Accuracy: {summ['accuracy']:.4f} | Balanced Acc: {summ['balanced_accuracy']:.4f} | "
            f"Macro-P: {summ['macro_precision']:.4f} | Macro-R: {summ['macro_recall']:.4f} | "
            f"Macro-F1: {summ['macro_f1']:.4f} | Weighted-F1: {summ['weighted_f1']:.4f}"
        )
        print(f"Majority Baseline Acc: {summ['majority_class_baseline_accuracy']:.4f} | NLL: {summ['nll']:.4f} | ECE: {summ['ece']:.4f}")
        print(summ["classification_report_text"])
        print("Confusion Matrix (rows=True, cols=Pred):")
        print(pd.DataFrame(summ["confusion_matrix"], index=ACTION_NAMES, columns=ACTION_NAMES).to_string())

    # 9. Build ConservativePolicy & RiskController
    risk_limits = RiskLimits.from_dict({**CONFIG["risk_limits"], "allow_short": CONFIG["allow_short"]})
    controller = RiskController(risk_limits)

    policy_config = {
        "model_name": "SP-05 Conservative Risk-Aware Policy",
        "policy_id": "SP-05",
        "version": "1.0.0",
        "artifact": "conservative_policy.pt",
        "seq_len": L,
        "allow_short": CONFIG["allow_short"],
        "temperature": temperature,
        "model": model_cfg,
        "action_space": {str(i): n for i, n in enumerate(ACTION_NAMES)},
        "label_source": "strategy_derived_proxy (conservative objective, decision-time)",
        "label_definition": {
            "information": "decision-time only (causal features, position, P&L, holding, portfolio state, profile)",
            "future_prices_used": False,
            "objective": CONFIG["objective"],
            "soft_limits": CONFIG["soft_limits"],
        },
        "authority": "This policy only PROPOSES actions. sp05_risk.RiskController has final authority over execution.",
        "risk_limits": risk_limits.to_dict(),
        "execution_convention": "decision at close of bar t, fill at open of bar t+1",
        "shared_schema": {
            "n_shared_sequence_features": N_BASE_FEATURES,
            "n_risk_features": len(EXTRA_FEATURE_COLUMNS),
            "n_shared_static_features": N_SHARED_STATIC,
            "n_portfolio_static_extras": N_STATIC - N_SHARED_STATIC,
            "adapter": "sp05_features.RiskFeatureAdapter",
        },
        "data": {
            "dataset": CONFIG["dataset_csv"],
            "symbols": list(raw_data),
            "benchmark": CONFIG["benchmark"],
            "train_end": str(d_train_end.date()),
            "val_end": str(d_val_end.date()),
            "source": "Indian Market Historical OHLCV (split/dividend adjusted)",
        },
        "training": {
            **CONFIG["train"],
            "seed": CONFIG["seed"],
            "best_epoch": int(best_epoch),
            "epochs_completed": int(len(history)),
            "training_duration_seconds": round(train_duration, 2),
            "class_weights": class_weights.tolist(),
            "n_train_samples": int(n_train),
        },
        "intended_use": "research and paper trading only; not financial advice",
    }

    policy = ConservativePolicy(net, policy_config, preprocessing, device)

    # 10. Portfolio Backtest on VAL and UNTOUCHED TEST
    print("\nRunning Portfolio Backtest (With Risk Gate vs Without Risk Gate)...")
    bt_results = {}
    curve_frames = []
    for sp in ("val", "test"):
        r_sp = eligible_rows(sp)
        res_gate = run_portfolio_backtest(panel, raw_mat, r_sp, bench_df, policy, controller, use_gate=True)
        res_nogate = run_portfolio_backtest(panel, raw_mat, r_sp, bench_df, policy, controller, use_gate=False)
        bm_bench = bh_metrics(res_gate["bench_ret"])
        bm_ew = bh_metrics(res_gate["ew_ret"])
        bt_results[sp] = {
            "policy_with_gate": res_gate["metrics"],
            "policy_without_gate": res_nogate["metrics"],
            "benchmark_buy_hold": bm_bench,
            "equal_weight_buy_hold": bm_ew,
            "gate_reason_counts": res_gate["gate_reason_counts"],
        }
        print(f"  [{sp.upper()}] With Gate: Ret={res_gate['metrics']['total_return']*100:.2f}%, MaxDD={res_gate['metrics']['max_drawdown']*100:.2f}%, Sharpe={res_gate['metrics']['sharpe']:.2f}, Overrides={res_gate['metrics']['gate_overrides']}")
        print(f"  [{sp.upper()}] No Gate  : Ret={res_nogate['metrics']['total_return']*100:.2f}%, MaxDD={res_nogate['metrics']['max_drawdown']*100:.2f}%, Sharpe={res_nogate['metrics']['sharpe']:.2f}")
        print(f"  [{sp.upper()}] NIFTY 50 : Ret={bm_bench['total_return']*100:.2f}%, MaxDD={bm_bench['max_drawdown']*100:.2f}%, Sharpe={bm_bench['sharpe']:.2f}")

        curve_frames.append(
            pd.DataFrame({
                "date": res_gate["dates"],
                "split": sp,
                "strategy_equity": res_gate["equity"],
                "no_gate_equity": res_nogate["equity"],
                "benchmark_equity": np.cumprod(1.0 + res_gate["bench_ret"]),
                "equal_weight_equity": np.cumprod(1.0 + res_gate["ew_ret"]),
                "gross_exposure": res_gate["exposure"],
                "portfolio_drawdown": res_gate["portfolio_drawdown"],
            })
        )

    curves_df = pd.concat(curve_frames, ignore_index=True)
    curves_df.to_csv(art_dir / "equity_curve.csv", index=False)

    # 11. Generate Plots
    print("\nGenerating Evaluation Graphs...")
    # Plot 1: Training & Validation Loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="Training Loss", marker="o", ms=4)
    ax.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss", marker="s", ms=4)
    ax.axvline(best_epoch, color="gray", ls="--", lw=1.2, label=f"Best Epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted Cross-Entropy Loss")
    ax.set_title("SP-05 Training vs Validation Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "training_validation_loss.png", dpi=150)
    plt.close(fig)

    # Plot 2: Training & Validation Accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["epoch"], history_df["train_acc"], label="Training Accuracy", marker="o", ms=4)
    ax.plot(history_df["epoch"], history_df["val_acc"], label="Validation Accuracy", marker="s", ms=4)
    ax.plot(history_df["epoch"], history_df["val_macro_f1"], label="Validation Macro-F1", ls=":", marker="^", ms=4)
    ax.axvline(best_epoch, color="gray", ls="--", lw=1.2, label=f"Best Epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("SP-05 Training vs Validation Accuracy & Macro-F1")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "training_validation_accuracy.png", dpi=150)
    plt.close(fig)

    # Plot 3: Combined Learning Curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    axes[0].axvline(best_epoch, color="gray", ls="--", label=f"Best ({best_epoch})")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(history_df["epoch"], history_df["train_acc"], label="Train Acc")
    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="Val Acc")
    axes[1].plot(history_df["epoch"], history_df["val_macro_f1"], label="Val Macro-F1")
    axes[1].axvline(best_epoch, color="gray", ls="--", label=f"Best ({best_epoch})")
    axes[1].set_title("Accuracy & Macro-F1 Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "learning_curves.png", dpi=150)
    plt.close(fig)

    # Plot 4: Confusion Matrix (Untouched Test Set)
    cm = np.array(test_summary["confusion_matrix"])
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(N_ACTIONS))
    ax.set_yticks(range(N_ACTIONS))
    ax.set_xticklabels(ACTION_NAMES)
    ax.set_yticklabels(ACTION_NAMES)
    ax.set_xlabel("Predicted Action")
    ax.set_ylabel("True Action")
    ax.set_title("SP-05 Test Set Confusion Matrix (Row-Normalized)")
    for i in range(N_ACTIONS):
        for j in range(N_ACTIONS):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)", ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Plot 5: Test Set Backtest Equity Curve
    test_curve = curves_df[curves_df["split"] == "test"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pd.to_datetime(test_curve["date"]), test_curve["strategy_equity"], label="SP-05 Policy + Risk Gate", lw=2)
    ax.plot(pd.to_datetime(test_curve["date"]), test_curve["no_gate_equity"], label="SP-05 Policy (No Gate)", lw=1.5, ls="--")
    ax.plot(pd.to_datetime(test_curve["date"]), test_curve["benchmark_equity"], label="NIFTY 50 Buy & Hold", lw=1.5, alpha=0.8)
    ax.plot(pd.to_datetime(test_curve["date"]), test_curve["equal_weight_equity"], label="Equal-Weight 16-Stock B&H", lw=1.2, alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of 1.0 Capital")
    ax.set_title("Untouched Test Split Backtest Equity Curves")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "backtest_equity.png", dpi=150)
    plt.close(fig)

    # 12. Save Standardized Artifacts
    print("\nSaving Standardized Model Artifacts...")
    pt_path = policy.save(art_dir)
    for mod in ("sp05_features.py", "sp05_risk.py", "sp05_model.py"):
        shutil.copy2(mod, art_dir / mod)

    evaluation_payload = {
        "project": "SP-05 Conservative Risk-Aware Policy",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "total_csv_rows_scanned": int(total_csv_rows),
        "universe_symbols": list(raw_data),
        "benchmark": CONFIG["benchmark"],
        "split_config": split_config,
        "trainable_parameters": int(n_params),
        "best_epoch": int(best_epoch),
        "epochs_completed": int(len(history)),
        "training_duration_seconds": round(train_duration, 2),
        "temperature": temperature,
        "validation_metrics": val_summary,
        "test_metrics": test_summary,
        "test_metrics_uncalibrated": {"nll": test_uncal["nll"], "ece": test_uncal["ece"]},
        "backtest": bt_results,
        "training_history": history,
    }
    (art_dir / "evaluation_metrics.json").write_text(json.dumps(to_py(evaluation_payload), indent=2))

    # 13. Deterministic Reload Verification (no CSV needed)
    print("\nRunning Reload & Determinism Verification...")
    reloaded = ConservativePolicy.load(art_dir, device)
    test_rows_sample = eligible_rows("test")[:64]
    win_sample = raw_mat[test_rows_sample[:, None] + offsets_np]
    state_ext = {"sequence": win_sample, "position": 0, "profile": (0.5, 0.5, 0.5)}
    out_orig = policy.predict(state_ext)
    out_rel = reloaded.predict(state_ext)
    assert np.allclose(out_orig["action_probs"], out_rel["action_probs"], atol=1e-6), "Reload probability mismatch!"
    assert (out_orig["action"] == out_rel["action"]).all(), "Reload action mismatch!"
    state_adapter = {
        "sequence": win_sample[:, :, :N_BASE_FEATURES],
        "risk_features": win_sample[:, :, N_BASE_FEATURES:],
        "position": 0,
        "profile": (0.5, 0.5, 0.5),
    }
    out_adapt = reloaded.predict(state_adapter)
    assert np.allclose(out_adapt["action_probs"], out_rel["action_probs"], atol=1e-6), "Adapter vs extended mismatch!"
    assert RiskController.from_policy_config(reloaded.config).limits.to_dict() == risk_limits.to_dict(), "RiskLimits round-trip failed!"
    print("ALL ROUND-TRIP & DETERMINISM CHECKS PASSED!")
    print(f"Total pipeline execution time: {time.time() - t_start:.1f}s")
    print(f"Artifacts saved to: {art_dir.resolve()}")


if __name__ == "__main__":
    main()

