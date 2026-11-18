# SP-05 Conservative Risk-Aware Trading Policy (`Conservation-model-stratym`)

A trained, calibrated, and deployment-ready PyTorch deep learning policy (`ConservativePolicyNet`) paired with a deterministic risk controller (`RiskController`) for risk-aware equity action selection (`HOLD`, `BUY`, `SELL`, `EXIT`).

> **Disclaimer:** Research and paper-trading system only. Labels are strategy-derived decision-time proxy labels, not human demonstrations or financial advice.

---

## Architecture Overview

1. **Causal Feature Engineering (`sp05_features.py`):**
   - **43 sequential market & risk features** (`seq_len = 30` daily bars): 30 shared market features (returns, RSI-14, MACD, EMA/SMA trends, ADX/DI, ATR/realized volatility, volume anomalies, candle geometry, relative strength vs benchmark) + 13 SP-05 risk features (252-bar ATR & realized-vol percentiles, 60/252-bar drawdowns, rolling max drawdown, gap ATR, abnormal return z-score, range z-score, volume anomaly z-score, Bollinger width, downside volatility, benchmark drawdown & vol percentile).
   - **12 static inputs:** Position one-hot (`flat`, `long`, `short`), trader profile (`risk_tolerance`, `patience`, `aggressiveness`), unrealized P&L, holding bars, portfolio drawdown, gross exposure, position size, and recent 20-bar loss.
2. **Neural Policy Network (`sp05_model.py` — `ConservativePolicyNet`, 51,540 parameters):**
   - 2-layer `GRU(43 -> 64, dropout=0.25)` over the 30-bar sequence
   - Static MLP `Linear(12 -> 16) + GELU`
   - Action head `Linear(80 -> 64) + GELU + Dropout(0.25) + Linear(64 -> 4)` with invalid-action masking (`-1e9`) and validation-fitted temperature scaling (`T = 1.2214`).
3. **Deterministic Safety Gate (`sp05_risk.py` — `RiskController`):**
   - Pure rule-based hard gate with final execution authority over neural proposals (enforcing position stop-loss, max holding bars, portfolio drawdown halt/flatten, volatility/gap/market-stress blocks, gross exposure caps, and minimum entry confidence).

---

## Training & Genuine Untouched Test Summary

Trained on 16 liquid Indian large-cap equities (`RELIANCE`, `TCS`, `HDFCBANK`, `INFY`, `ICICIBANK`, `HINDUNILVR`, `ITC`, `SBIN`, `BHARTIARTL`, `KOTAKBANK`, `LT`, `AXISBANK`, `TITAN`, `TATASTEEL`, `SUNPHARMA`, `WIPRO`) with `NIFTY 50` as benchmark (`2008-11-20` to `2026-10-23`, seed `42`).

- **Strict Chronological 80:20 Split (`strict_window_isolation=True`, zero leakage):**
  - **Train:** 3,186 dates (72.00% of dates) / 151,536 samples (`2009-01-05` to `2021-10-26`) — scaler & class weights fit exclusively here
  - **Validation:** 354 dates (8.00% of dates) / 15,600 samples (`2021-12-09` to `2023-03-28`) — early stopping & temperature calibration only
  - **Untouched Test:** 885 dates (20.00% of dates) / 41,052 samples (`2023-05-16` to `2026-10-23`) — evaluated once after training

| Metric (Untouched 20% Test Set, `n = 41,052`) | Value |
|---|---:|
| **Test Accuracy** | **96.81%** (vs 53.67% majority baseline) |
| **Balanced Accuracy** | **89.12%** |
| **Weighted Precision / Recall / F1** | **0.9700 / 0.9681 / 0.9687** |
| **Macro Precision / Recall / F1** | **0.7058 / 0.8912 / 0.7608** |
| **Calibrated NLL / ECE** | **0.0798 / 0.0024** |
| **Test Backtest Max Drawdown (Policy + Gate vs NIFTY 50)** | **-6.50% vs -15.77%** |

See [`TRAINING_REPORT.md`](TRAINING_REPORT.md) and [`sp05_conservative_artifacts/evaluation_metrics.json`](sp05_conservative_artifacts/evaluation_metrics.json) for full per-class metrics, confusion matrix, backtest curves, and limitations.

---

## Installation

```bash
pip install -r requirements.txt
```

Works on both **CPU** and **CUDA GPU** without retraining and **without requiring the original training CSV dataset**.

---

## Running Predictions

### Quick Self-Contained Verification (No CSV Required)
```bash
python predict.py --demo
```

### Inference from a JSON State File
```bash
python predict.py --input sample_state.json
```

### Python API Usage
```python
from predict import load_inference_engine, predict_action

# Loads weights, preprocessing scaler, and config from sp05_conservative_artifacts/conservative_policy.pt
policy, controller = load_inference_engine("./sp05_conservative_artifacts", device="cpu")

# Build state from your own OHLCV DataFrames (stock + benchmark, >= 320 daily bars, columns: Open, High, Low, Close, Volume)
state = policy.state_from_ohlcv(
    stock_ohlcv_df,
    benchmark_ohlcv_df,
    position=0,                     # -1 (short), 0 (flat), +1 (long)
    profile=(0.5, 0.5, 0.5),        # (risk_tolerance, patience, aggressiveness)
    portfolio_drawdown=0.01,
    gross_exposure=0.20,
    recent_loss=0.00,
)

result = predict_action(policy, controller, state)
print(result)
```

### Input / Output Contract
- **Input (`state` dict):**
  - `sequence`: float32 array of shape `(30, 43)` (raw extended features) **or** `(30, 30)` + `risk_features` of shape `(30, 13)`. Scaling (train median imputation -> z-score -> clip `±5`) is applied internally by `ConservativePolicy`.
  - `position`: `int` in `{-1, 0, 1}` (default `0`).
  - `profile`: `[risk_tolerance, patience, aggressiveness]` in `[0, 1]^3` (default `(0.5, 0.5, 0.5)`).
  - Optional scalars: `unrealized_pnl`, `holding_bars`, `portfolio_drawdown`, `gross_exposure`, `position_size`, `recent_loss`.
- **Output (`dict`):**
  - `proposed_action` (`HOLD` | `BUY` | `SELL` | `EXIT`), `action_probabilities`, `calibrated_confidence`, `entropy`, `valid_actions`, `final_action` (post-`RiskController`), `gate_overridden`, `gate_reasons`.

---

## Dataset Privacy & Availability

The raw historical training dataset (`indian-market-historical-ohlcv.csv`, ~942 MB) is **intentionally excluded** from this GitHub repository (via `.gitignore`) and preserved locally only. All saved model weights, scalers, and configurations inside `sp05_conservative_artifacts/` are completely self-contained for deployment inference without access to the training CSV. To retrain locally, place `indian-market-historical-ohlcv.csv` in the repository root and run `python train_conservative_policy.py`.

---

## Repository Structure

```text
├── sp05_features.py                        # Causal 43-feature & 12-static-input engineering
├── sp05_model.py                           # ConservativePolicyNet (GRU) & ConservativePolicy wrapper
├── sp05_risk.py                            # Deterministic RiskController hard safety gate
├── predict.py                              # Standalone CLI & Python inference entry point (no CSV needed)
├── train_conservative_policy.py            # Reproducible training, evaluation & backtest pipeline
├── requirements.txt                        # Deployment Python dependencies
├── TRAINING_REPORT.md                      # Detailed training, split, evaluation & backtest report
├── ERROR_LOG.md                            # Debugging & environment resolution log
├── .gitignore                              # Excludes raw CSV datasets, checkpoints, caches, secrets
└── sp05_conservative_artifacts/
    ├── conservative_policy.pt              # Self-contained trained model weights + scaler + config
    ├── conservative_config.json            # Full model, risk limits, and calibration config
    ├── feature_schema.json                 # Machine-readable I/O contract
    ├── preprocessing.pkl                   # Train-fitted median/mean/std/clip scaler
    ├── split_config.json                   # Exact chronological 72/8/20 split metadata (seed 42)
    ├── evaluation_metrics.json             # Complete validation, untouched test, and backtest metrics
    ├── training_history.csv                # Per-epoch loss, accuracy, macro-F1, and learning rate
    ├── equity_curve.csv                    # Validation & test backtest equity curves
    └── figures/
        ├── training_validation_loss.png
        ├── training_validation_accuracy.png
        ├── learning_curves.png
        ├── confusion_matrix.png
        └── backtest_equity.png
```
