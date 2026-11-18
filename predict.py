"""Standalone Inference Script for SP-05 Conservative Risk-Aware Policy.

Loads the trained model and preprocessing parameters directly from
`sp05_conservative_artifacts/conservative_policy.pt` (no training CSV required),
runs calibrated action prediction, and passes the proposal through the
deterministic `RiskController` hard safety gate.

Usage:
    python predict.py --demo
    python predict.py --input sample_state.json
    python predict.py --model-dir ./sp05_conservative_artifacts --demo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sp05_features import (
    ACTION_NAMES,
    DEFAULT_PROFILE,
    FEATURE_COLUMNS,
    N_ACTIONS,
    N_BASE_FEATURES,
    N_FEATURES,
    N_STATIC,
    WARMUP_BARS,
)
from sp05_model import ConservativePolicy
from sp05_risk import RiskController


def load_inference_engine(model_dir: str | Path = "./sp05_conservative_artifacts", device: str | None = None):
    """Load the saved ConservativePolicy and its matching deterministic RiskController."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = ConservativePolicy.load(model_dir, device=dev)
    controller = RiskController.from_policy_config(policy.config)
    return policy, controller


def predict_action(policy: ConservativePolicy, controller: RiskController, state: dict) -> dict:
    """Run policy proposal + RiskController hard gate review for a single state."""
    proposal = policy.predict(state)
    gate_decision = controller.review(proposal, state, FEATURE_COLUMNS)

    probs_map = {
        ACTION_NAMES[i]: round(float(proposal["action_probs"][i]), 6)
        for i in range(N_ACTIONS)
    }
    return {
        "proposed_action_id": int(proposal["action"]),
        "proposed_action": proposal["action_name"],
        "action_probabilities": probs_map,
        "calibrated_confidence": round(float(proposal["confidence"]), 6),
        "entropy": round(float(proposal["entropy"]), 6),
        "valid_actions": [ACTION_NAMES[i] for i, ok in enumerate(proposal["valid_actions"]) if ok],
        "final_action_id": int(gate_decision["action"]),
        "final_action": ACTION_NAMES[int(gate_decision["action"])],
        "gate_overridden": bool(gate_decision["overridden"]),
        "gate_reasons": gate_decision["reasons"],
    }


def build_demo_ohlcv(n_bars: int = 340, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic OHLCV series (no external CSV) purely to verify inference pipeline."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_bars)

    def _make_series(start_px: float, drift: float, vol: float) -> pd.DataFrame:
        rets = rng.normal(drift, vol, size=n_bars)
        close = start_px * np.cumprod(1.0 + rets)
        open_ = close * (1.0 + rng.normal(0.0, vol * 0.25, size=n_bars))
        spread = np.abs(rng.normal(0.0, vol * 0.6, size=n_bars))
        high = np.maximum(open_, close) * (1.0 + spread)
        low = np.minimum(open_, close) * np.maximum(0.01, 1.0 - spread)
        vol_arr = rng.integers(500_000, 5_000_000, size=n_bars).astype(float)
        return pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol_arr},
            index=dates,
        )

    stock_df = _make_series(start_px=1500.0, drift=0.0006, vol=0.012)
    bench_df = _make_series(start_px=24000.0, drift=0.0004, vol=0.009)
    return stock_df, bench_df


def run_demo(model_dir: str | Path) -> None:
    policy, controller = load_inference_engine(model_dir)
    print(f"Loaded SP-05 ConservativePolicy from: {Path(model_dir).resolve()}")
    print(f"Device: {policy.device} | seq_len={policy.seq_len} | temperature={policy.temperature:.4f}")
    print(f"Expected sequence shape: (B, {policy.seq_len}, {N_FEATURES}) [or ({policy.seq_len}, {N_BASE_FEATURES}) + risk_features ({policy.seq_len}, {N_FEATURES - N_BASE_FEATURES})]")
    print(f"Expected static dims   : {N_STATIC} | Actions: {ACTION_NAMES}\n")

    stock_df, bench_df = build_demo_ohlcv(n_bars=WARMUP_BARS + policy.seq_len + 20)

    scenarios = [
        {
            "name": "Scenario 1: Flat position, calm portfolio, balanced profile",
            "kwargs": {"position": 0, "profile": DEFAULT_PROFILE, "portfolio_drawdown": 0.01, "gross_exposure": 0.20, "recent_loss": 0.00},
        },
        {
            "name": "Scenario 2: Long position in profit (+3.5%, 10 bars held)",
            "kwargs": {"position": 1, "profile": (0.2, 0.8, 0.2), "unrealized_pnl": 0.035, "holding_bars": 10, "portfolio_drawdown": 0.01, "gross_exposure": 0.30, "position_size": 0.10, "recent_loss": 0.00},
        },
        {
            "name": "Scenario 3: Long position breaching hard stop-loss (-6.5%, gate forced EXIT)",
            "kwargs": {"position": 1, "profile": DEFAULT_PROFILE, "unrealized_pnl": -0.065, "holding_bars": 15, "portfolio_drawdown": 0.04, "gross_exposure": 0.35, "position_size": 0.10, "recent_loss": 0.02},
        },
        {
            "name": "Scenario 4: Flat position during severe portfolio drawdown (10%, gate halts new entry)",
            "kwargs": {"position": 0, "profile": (0.8, 0.3, 0.8), "portfolio_drawdown": 0.10, "gross_exposure": 0.10, "recent_loss": 0.07},
        },
    ]

    for sc in scenarios:
        state = policy.state_from_ohlcv(stock_df, bench_df, **sc["kwargs"])
        seq_shape = state["sequence"].shape
        risk_shape = state["risk_features"].shape
        res = predict_action(policy, controller, state)
        print(f"=== {sc['name']} ===")
        print(f"  Input shapes  : sequence={seq_shape}, risk_features={risk_shape}")
        print(f"  Proposed      : {res['proposed_action']} (conf={res['calibrated_confidence']:.4f}, entropy={res['entropy']:.4f})")
        print(f"  Probabilities : {res['action_probabilities']}")
        print(f"  Gate Final    : {res['final_action']} (overridden={res['gate_overridden']}, reasons={res['gate_reasons']})\n")

    # Batch verification
    s0 = policy.state_from_ohlcv(stock_df, bench_df, position=0)
    seq_full = np.concatenate([s0["sequence"], s0["risk_features"]], axis=1)  # (30, 43)
    batch_seq = np.stack([seq_full, seq_full, seq_full], axis=0)              # (3, 30, 43)
    batch_out = policy.predict({"sequence": batch_seq, "position": [0, 1, -1], "profile": DEFAULT_PROFILE})
    assert batch_out["action_probs"].shape == (3, N_ACTIONS), f"Unexpected batch output shape: {batch_out['action_probs'].shape}"
    assert np.allclose(batch_out["action_probs"].sum(axis=1), 1.0, atol=1e-5), "Probabilities do not sum to 1!"
    # Determinism check
    batch_out2 = policy.predict({"sequence": batch_seq, "position": [0, 1, -1], "profile": DEFAULT_PROFILE})
    assert np.array_equal(batch_out["action_probs"], batch_out2["action_probs"]), "Inference is not deterministic!"
    print(f"Batch verification PASSED: input={batch_seq.shape} -> output_probs={batch_out['action_probs'].shape}, actions={batch_out['action_name']}, deterministic=True.")


def run_from_json(model_dir: str | Path, json_path: str | Path) -> None:
    policy, controller = load_inference_engine(model_dir)
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    result = predict_action(policy, controller, payload)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="SP-05 Conservative Policy Inference CLI")
    parser.add_argument("--model-dir", default="./sp05_conservative_artifacts", help="Directory containing conservative_policy.pt")
    parser.add_argument("--demo", action="store_true", help="Run self-contained demo inference without any CSV")
    parser.add_argument("--input", type=str, default=None, help="Path to JSON file describing a single inference state")
    args = parser.parse_args()

    if args.input:
        run_from_json(args.model_dir, args.input)
    else:
        run_demo(args.model_dir)


if __name__ == "__main__":
    main()

