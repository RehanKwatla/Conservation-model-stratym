# SP-05 Conservative Risk-Aware Policy — Training & Evaluation Report

## 1. Project Overview
- **Model:** SP-05 Conservative Risk-Aware Policy (`ConservativePolicyNet` + `ConservativePolicy` + deterministic `RiskController`)
- **Origin Preserved:** Extracted verbatim from `05_conservative_policy.ipynb` (`sp05_features.py`, `sp05_risk.py`, `sp05_model.py`). Original notebook and original dataset remain completely unmodified locally.
- **Task:** 4-class position-conditioned action classification (`HOLD=0`, `BUY=1`, `SELL=2`, `EXIT=3`) under a decision-time conservative risk-aware objective, paired with a deterministic hard safety gate (`RiskController`).
- **Framework & Hardware:** PyTorch `2.6.0+cu124` on `NVIDIA T1000 8GB` GPU (Python `3.11.2`).

## 2. Dataset & Preprocessing
- **Local Dataset File:** `indian-market-historical-ohlcv.csv` (941.7 MB, preserved locally and excluded from GitHub)
- **Raw Dataset Scope:** 7,967,459 rows across 2,704 Indian market symbols (`stocks`, `indices`, `etfs`, `forex`, `commodities`, 1991-01-02 to 2026-10-25).
- **Selected Training Universe:**
  - **Benchmark:** `NIFTY 50` (4,684 daily bars: `2007-09-17` to `2026-10-23`)
  - **16 Liquid Large-Cap Equities:** `RELIANCE`, `TCS`, `HDFCBANK`, `INFY`, `ICICIBANK`, `HINDUNILVR`, `ITC`, `SBIN`, `BHARTIARTL`, `KOTAKBANK`, `LT`, `AXISBANK`, `TITAN`, `TATASTEEL`, `SUNPHARMA`, `WIPRO` (119,466 extracted bars total).
- **Cleaning & Adjustment:**
  - Applied split/dividend adjustment ratio `adj_close / close` to `Open, High, Low, Close` (matching `yfinance` `auto_adjust=True`).
  - Dropped 10 corrupted/missing-price rows across the universe (`RELIANCE: 1, KOTAKBANK: 1, LT: 1, AXISBANK: 1, TITAN: 1, SUNPHARMA: 5`); 0 duplicate dates found.
- **Causal Feature Engineering (43 Sequence Features + 12 Static Inputs):**
  - 30 base market features (`ret_*`, `rsi_14`, `macd_*`, EMA/SMA ratios, `adx_14`, `di_diff`, `atr_pct`, `rv_20/60`, volume ratios, candle shape, relative strength vs `NIFTY 50`).
  - 13 SP-05 risk features (`atr_pct_pctile_252`, `rv20_pctile_252`, `dd_from_high_60/252`, `maxdd_60`, `gap_abs_atr`, `abnormal_ret_z`, `range_z`, `vol_anomaly_z`, `bb_width_20`, `down_vol_20`, `bench_dd_60`, `bench_rv20_pctile_252`).
  - Causality verified via future-perturbation test on `RELIANCE` (`atol=1e-12`, zero lookahead leakage).
  - After 290-bar warmup per symbol, the feature panel contains **70,788 bars** across **4,425 trading dates** (`2008-11-20` to `2026-10-23`).
- **Leakage-Free Scaling:**
  - Median imputation, mean, and standard deviation were fit **exclusively on the 50,976 TRAIN rows (`<= 2021-10-26`)**, then applied to validation and test splits with clipping to `[-5.0, +5.0]`.

## 3. Strict Chronological 80:20 Split (72% / 8% / 20%)
- **Random Seed:** `42`
- **Method:** Chronological calendar-date split with `strict_window_isolation=True` (30-bar rolling windows never cross split boundaries).
- **Date-Level Split (4,425 panel trading dates):**
  - **Train:** `3,186 dates (72.00%)` — `<= 2021-10-26`
  - **Validation:** `354 dates (8.00%)` — `2021-10-27` to `2023-03-28`
  - **Development (Train + Val):** `3,540 dates (80.00%)`
  - **Untouched Test:** `885 dates (20.00%)` — `>= 2023-03-29`

| Split | Decision Date Range | Exact Samples | % of Samples | HOLD | BUY | SELL | EXIT |
|---|---|---:|---:|---:|---:|---:|---:|
| **Train** | `2009-01-05` -> `2021-10-26` | 151,536 | 72.79% | 67,220 | 2,941 | 7 | 81,368 |
| **Validation** | `2021-12-09` -> `2023-03-28` | 15,600 | 7.49% | 6,965 | 201 | 1 | 8,433 |
| **Dev (Train + Val)** | `2009-01-05` -> `2023-03-28` | **167,136** | **80.28%** | 74,185 | 3,142 | 8 | 89,801 |
| **Test (Untouched)** | `2023-05-16` -> `2026-10-23` | **41,052** | **19.72%** | 18,417 | 598 | 3 | 22,034 |
| **Total** | `2009-01-05` -> `2026-10-23` | **208,188** | **100.00%** | 92,602 | 3,740 | 11 | 111,835 |

*Note: Sample percentages differ by ~0.28% from exact 72/8/20 date ratios because `strict_window_isolation=True` excludes the first 29 bars of each split so no 30-bar sequence window overlaps the previous split.*

## 4. Model Architecture & Training Configuration
- **Architecture (`ConservativePolicyNet`):**
  - Sequence Encoder: 2-layer `GRU(input_size=43, hidden_size=64, dropout=0.25)`
  - Static Branch: `Linear(12 -> 16) + GELU`
  - Action Head: `Linear(80 -> 64) + GELU + Dropout(0.25) + Linear(64 -> 4)` with invalid-action logit masking (`-1e9`)
  - **Trainable Parameters:** `51,540`
- **Training Setup:**
  - Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`
  - Scheduler: `ReduceLROnPlateau(factor=0.5, patience=2)`
  - Loss: Inverse-frequency class-weighted cross-entropy (computed on TRAIN only: `HOLD=0.751, BUY=17.154, SELL=26.634, EXIT=0.620`)
  - Regularization: Input Gaussian noise `std=0.02`, gradient clipping `1.0`, dropout `0.25`, early stopping `patience=7`
  - Batch Size: `512` | Max Epochs: `40`
- **Execution Results:**
  - **Epochs Completed:** `40` (early stopping triggered at epoch 40 after 7 epochs without val improvement)
  - **Best Checkpoint:** Epoch `33` (`val_loss = 0.0721`, `val_acc = 97.38%`)
  - **Training Duration:** `99.1 seconds` on NVIDIA T1000 8GB GPU
  - **Temperature Calibration (Validation Only):** `T = 1.2214`

## 5. Final Evaluation on Untouched Test Dataset (41,052 Samples)

### 5.1 Overall Classification Metrics
| Metric | Validation (15,600) | Untouched Test (41,052) |
|---|---:|---:|
| **Accuracy** | 97.38% | **96.81%** |
| **Majority-Class Baseline** | 54.06% | 53.67% |
| **Balanced Accuracy** | 73.15% | **89.12%** |
| **Weighted Precision** | 0.9754 | **0.9700** |
| **Weighted Recall** | 0.9738 | **0.9681** |
| **Weighted F1-Score** | 0.9743 | **0.9687** |
| **Macro Precision** | 0.6641 | **0.7058** |
| **Macro Recall** | 0.7315 | **0.8912** |
| **Macro F1-Score** | 0.6919 | **0.7608** |
| **Calibrated NLL** | 0.0695 | **0.0798** |
| **Calibrated ECE** | 0.0045 | **0.0024** |

### 5.2 Per-Class Test Report
| Action Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| **HOLD** | 0.9683 | 0.9603 | 0.9643 | 18,417 |
| **BUY** | 0.6753 | 0.9632 | 0.7939 | 598 |
| **SELL** | 0.2000 | 0.6667 | 0.3077 | 3 |
| **EXIT** | 0.9796 | 0.9748 | 0.9772 | 22,034 |

### 5.3 Untouched Test Confusion Matrix (Rows = True, Cols = Predicted)
| True \ Pred | HOLD | BUY | SELL | EXIT |
|---|---:|---:|---:|---:|
| **HOLD** | 17,685 | 277 | 8 | 447 |
| **BUY** | 22 | 576 | 0 | 0 |
| **SELL** | 1 | 0 | 2 | 0 |
| **EXIT** | 556 | 0 | 0 | 21,478 |

### 5.4 Portfolio Backtest Summary (Decision at Close `t`, Fill at Open `t+1`, 5 bps fee + 5 bps slippage)
| Strategy | Val Return | Val Max DD | Val Sharpe | Test Return | Test Max DD | Test Sharpe | Test Gate Overrides |
|---|---:|---:|---:|---:|---:|---:|---:|
| **SP-05 Policy + Risk Gate** | -3.87% | -5.93% | -0.75 | +1.72% | **-6.50%** | 0.13 | 241 |
| **SP-05 Policy (No Gate)** | -2.79% | -5.05% | -0.49 | +2.11% | -6.18% | 0.16 | — |
| **NIFTY 50 Buy & Hold** | -3.23% | -16.47% | -0.16 | +23.59% | -15.77% | 0.50 | — |

### 5.5 Honest Interpretation & Limitations
1. **High Fidelity to Conservative Policy:** The model achieves 96.81% accuracy (vs 53.67% majority baseline), 0.9687 weighted F1, and exceptional probability calibration (`ECE = 0.0024`) on nearly 3.5 years of unseen future market data (2023–2026).
2. **Rare `SELL` Class:** Because the conservative objective penalizes shorts (`short_penalty=0.2`) and Indian large-cap equities experienced a secular uptrend, only 7 `SELL` training samples, 1 validation sample, and 3 test samples occur. Consequently, `SELL` precision (0.20) is noisy due to extreme rarity.
3. **Conservative Return vs. Drawdown Tradeoff:** As a defensive policy designed to preserve capital during drawdowns, SP-05 cut maximum drawdown by more than half vs `NIFTY 50` (-6.50% vs -15.77% on Test; -5.93% vs -16.47% on Val), at the expected cost of lagging a strong bull rally in raw return (+1.72% vs +23.59%).
4. **Proxy Labels:** Labels are decision-time strategy-derived proxy labels, not human trader demonstrations. Intended for research and paper trading only.

## 6. Saved Deliverables & Graphs
- **Final Self-Contained Model:** `sp05_conservative_artifacts/conservative_policy.pt`
- **Best Training Checkpoint (Local):** `checkpoints/best_checkpoint.pt`
- **Preprocessing & Config:** `sp05_conservative_artifacts/preprocessing.pkl`, `sp05_conservative_artifacts/conservative_config.json`, `sp05_conservative_artifacts/feature_schema.json`, `sp05_conservative_artifacts/split_config.json`
- **Metrics & History:** `sp05_conservative_artifacts/evaluation_metrics.json`, `sp05_conservative_artifacts/training_history.csv`, `sp05_conservative_artifacts/equity_curve.csv`
- **Graphs:**
  - `sp05_conservative_artifacts/figures/training_validation_loss.png`
  - `sp05_conservative_artifacts/figures/training_validation_accuracy.png`
  - `sp05_conservative_artifacts/figures/learning_curves.png`
  - `sp05_conservative_artifacts/figures/confusion_matrix.png`
  - `sp05_conservative_artifacts/figures/backtest_equity.png`
- **Inference Script:** `predict.py` (verified on both CUDA and CPU with zero dependency on `indian-market-historical-ohlcv.csv`).

