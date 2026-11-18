"""SP-05 deterministic risk controller (the hard gate).

This module contains NO neural network and NO learned parameters. The policy (sp05_model.ConservativePolicy) only
PROPOSES an action; ``RiskController`` has final authority over what may execute. Rules use RAW (unscaled) feature
values and portfolio state, are pure functions, and are identical in backtests and in live/paper use.

Rules
-----
Entry blocks (a proposed BUY/SELL becomes HOLD):
  short_not_allowed, drawdown_halt, recent_loss_halt, volatility_prohibited, abnormal_gap_or_move, market_stress,
  gross_exposure_limit, low_confidence_entry, risk_data_missing (fail-safe: unknown risk -> no new entries)
Forced exits (any position becomes EXIT, whatever was proposed):
  position_stop_loss, max_holding_period, drawdown_flatten
De-risking proposals (EXIT) are never blocked. Invalid proposals for the current position become HOLD (invalid_action).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from sp05_features import BUY, EXIT, HOLD, SELL, valid_action_mask

REASON_NAMES = [
    "invalid_action", "short_not_allowed", "position_stop_loss", "max_holding_period", "drawdown_flatten",
    "drawdown_halt", "recent_loss_halt", "volatility_prohibited", "abnormal_gap_or_move", "market_stress",
    "gross_exposure_limit", "low_confidence_entry", "risk_data_missing",
]
FORCED_EXIT_REASONS = ["position_stop_loss", "max_holding_period", "drawdown_flatten"]
REQUIRED_FEATURES = ["atr_pct_pctile_252", "rv20_pctile_252", "gap_abs_atr", "abnormal_ret_z", "bench_dd_60"]


@dataclass
class RiskLimits:
    allow_short: bool = True
    max_position_fraction: float = 0.10     # entry size cap, fraction of equity
    max_gross_exposure: float = 0.60        # sum |notional| / equity after the entry
    position_stop_loss: float = 0.05        # hard per-position loss limit -> forced EXIT
    max_holding_bars: int = 120             # hard time limit -> forced EXIT (0 disables)
    halt_drawdown: float = 0.08             # portfolio drawdown (below the trailing ~120-bar high): no new entries
    flatten_drawdown: float = 0.15          # portfolio drawdown: force EXIT everywhere
    recent_loss_halt: float = 0.06          # ~20-bar equity loss: no new entries
    max_vol_percentile: float = 0.90        # no new entries when ATR or realised-vol percentile is above this
    max_abs_gap_atr: float = 3.0            # no new entries after a gap larger than this many ATRs
    max_abnormal_ret_z: float = 4.0         # no new entries after a move this many sigmas abnormal
    market_stress_dd: float = 0.12          # no new entries when the benchmark is this far below its 60-bar high
    min_entry_confidence: float = 0.50      # no new entries when the proposing policy's confidence is below this

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RiskLimits":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RiskController:
    """Deterministic gate: final authority over execution."""

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()

    @classmethod
    def from_policy_config(cls, config: dict) -> "RiskController":
        return cls(RiskLimits.from_dict(config.get("risk_limits", {})))

    # ------------------------------------------------------------------ batch implementation
    def review_batch(self, proposed, confidence, position, unrealized_pnl, holding_bars, features: dict, portfolio: dict):
        """Review one proposal per row.

        proposed, confidence, position, unrealized_pnl, holding_bars : arrays (B,)
        features  : dict of RAW latest-bar feature arrays (B,) containing REQUIRED_FEATURES
        portfolio : dict with scalars  portfolio_drawdown (>=0), gross_exposure, entry_fraction (planned entry size)

        Returns (final_actions (B,), reasons (B, len(REASON_NAMES)) bool).
        """
        lim = self.limits
        proposed = np.asarray(proposed).astype(int).reshape(-1)
        n = proposed.size
        conf = np.asarray(confidence, dtype=float).reshape(-1)
        pos = np.asarray(position).astype(int).reshape(-1)
        pnl = np.asarray(unrealized_pnl, dtype=float).reshape(-1)
        hold = np.asarray(holding_bars, dtype=float).reshape(-1)
        R = {name: np.zeros(n, dtype=bool) for name in REASON_NAMES}
        final = proposed.copy()

        # 1) proposals that are impossible for the current position never execute
        valid = valid_action_mask(pos, lim.allow_short)
        bad = ~valid[np.arange(n), final]
        R["short_not_allowed"] = bad & (final == SELL) & (pos == 0) & (not lim.allow_short)
        R["invalid_action"] = bad & ~R["short_not_allowed"]
        final[bad] = HOLD

        # 2) entry blocks
        pdd = float(portfolio.get("portfolio_drawdown", 0.0)); gross = float(portfolio.get("gross_exposure", 0.0))
        rloss = float(portfolio.get("recent_loss", 0.0))
        entry = (final == BUY) | (final == SELL)
        with np.errstate(invalid="ignore"):
            vol_pct = np.maximum(np.asarray(features["atr_pct_pctile_252"], float), np.asarray(features["rv20_pctile_252"], float))
            known = np.all([np.isfinite(np.asarray(features[k], float)) for k in REQUIRED_FEATURES], axis=0)
            R["risk_data_missing"] = entry & ~known
            R["drawdown_halt"] = entry & (pdd >= lim.halt_drawdown)
            R["recent_loss_halt"] = entry & (rloss >= lim.recent_loss_halt)
            R["volatility_prohibited"] = entry & (vol_pct > lim.max_vol_percentile)
            R["abnormal_gap_or_move"] = entry & ((np.asarray(features["gap_abs_atr"], float) > lim.max_abs_gap_atr)
                                                 | (np.abs(np.asarray(features["abnormal_ret_z"], float)) > lim.max_abnormal_ret_z))
            R["market_stress"] = entry & (np.asarray(features["bench_dd_60"], float) <= -lim.market_stress_dd)
            R["low_confidence_entry"] = entry & (conf < lim.min_entry_confidence)
        blocked = np.zeros(n, dtype=bool)
        for k in ("risk_data_missing", "drawdown_halt", "recent_loss_halt", "volatility_prohibited", "abnormal_gap_or_move",
                  "market_stress", "low_confidence_entry"):
            blocked |= R[k]
        # gross-exposure budget, most confident entries first
        running = gross; frac = float(portfolio.get("entry_fraction", lim.max_position_fraction))
        frac = min(frac, lim.max_position_fraction)
        for i in np.argsort(-conf):
            if entry[i] and not blocked[i]:
                if running + frac <= lim.max_gross_exposure + 1e-12:
                    running += frac
                else:
                    R["gross_exposure_limit"][i] = True; blocked[i] = True
        final[blocked] = HOLD

        # 3) forced exits for held positions (overrule HOLD and any proposal other than EXIT)
        held = pos != 0
        with np.errstate(invalid="ignore"):
            R["position_stop_loss"] = held & (pnl <= -lim.position_stop_loss)
            R["max_holding_period"] = held & (lim.max_holding_bars > 0) & (hold >= lim.max_holding_bars)
        R["drawdown_flatten"] = held & (pdd >= lim.flatten_drawdown)
        forced = R["position_stop_loss"] | R["max_holding_period"] | R["drawdown_flatten"]
        final[forced] = EXIT
        reasons = np.stack([R[k] for k in REASON_NAMES], axis=1)
        return final, reasons

    # ------------------------------------------------------------------ single-state convenience
    def review(self, proposal: dict, state: dict, feature_names: list) -> dict:
        """Review one policy output. ``state['sequence']`` must contain the RAW features (all 43 columns, or the 30 shared +
        'risk_features'); the latest bar supplies the risk features. Returns a dict with the final action and reasons."""
        seq = np.asarray(state["sequence"], dtype=float)
        extra = state.get("risk_features")
        last = seq[-1] if seq.ndim == 2 else seq[0, -1]
        if extra is not None:
            ex = np.asarray(extra, dtype=float); last = np.concatenate([last, ex[-1] if ex.ndim == 2 else ex[0, -1]])
        feats = {name: np.array([last[i]]) for i, name in enumerate(feature_names) if i < last.size}
        port = {"portfolio_drawdown": state.get("portfolio_drawdown", 0.0) or 0.0, "gross_exposure": state.get("gross_exposure", 0.0) or 0.0,
                "recent_loss": state.get("recent_loss", 0.0) or 0.0}
        final, reasons = self.review_batch([proposal["action"]], [proposal.get("confidence", 1.0)], [state.get("position", 0)],
                                           [state.get("unrealized_pnl", 0.0) or 0.0], [state.get("holding_bars", 0) or 0], feats, port)
        names = [r for r, on in zip(REASON_NAMES, reasons[0]) if on]
        return {"action": int(final[0]), "proposed_action": int(proposal["action"]), "overridden": bool(final[0] != proposal["action"]), "reasons": names}