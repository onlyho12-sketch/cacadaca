"""Isolated 120 Hz curved-contact safety supervisor for Gate F10-G3.

This module is Isaac-free.  It emits a bounded force request and a retract
velocity request; a later isolated PhysX adapter may apply those requests.
It does not modify the shared environment or contact controller.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubstepSafetyConfig:
    hard_force_n: float = 14.0
    predictive_guard_n: float = 9.0
    prediction_horizon_substeps: float = 2.0
    release_raw_force_n: float = 7.0
    release_filtered_force_n: float = 8.0
    release_substeps: int = 3
    retract_velocity_m_s: float = 0.020
    retract_force_floor_n: float = 0.0

    def validate(self) -> None:
        if not (0.0 <= self.retract_force_floor_n <= self.release_raw_force_n
                < self.predictive_guard_n < self.hard_force_n):
            raise ValueError("invalid force thresholds")
        if self.release_filtered_force_n >= self.predictive_guard_n:
            raise ValueError("release filter threshold must be below guard")
        if self.prediction_horizon_substeps < 1.0 or self.release_substeps < 1:
            raise ValueError("invalid prediction/release horizon")
        if self.retract_velocity_m_s <= 0.0:
            raise ValueError("retract velocity must be positive")


@dataclass
class SubstepSafetyState:
    previous_raw_force_n: np.ndarray
    retract_latched: np.ndarray
    safe_streak: np.ndarray

    @classmethod
    def zeros(cls, count: int) -> "SubstepSafetyState":
        if count <= 0:
            raise ValueError("count must be positive")
        return cls(
            previous_raw_force_n=np.zeros(count, dtype=np.float64),
            retract_latched=np.zeros(count, dtype=bool),
            safe_streak=np.zeros(count, dtype=np.int64),
        )


def supervise_substep(
    parent_force_command_n,
    raw_force_n,
    filtered_force_n,
    pad_gap_m,
    *,
    surface_kind: str,
    state: SubstepSafetyState,
    cfg: SubstepSafetyConfig = SubstepSafetyConfig(),
) -> dict[str, np.ndarray]:
    """Return a fail-closed retract request without exceeding the parent force.

    Positive retract velocity increases clearance.  The supervisor predicts
    the next short-horizon raw force from the authoritative 120 Hz samples.
    ``pad_gap_m`` is included in the contract and diagnostics; force prediction
    remains authoritative because absolute gap depends on surface geometry.
    """
    cfg.validate()
    parent, raw, filtered, gap = np.broadcast_arrays(
        np.asarray(parent_force_command_n, dtype=np.float64),
        np.asarray(raw_force_n, dtype=np.float64),
        np.asarray(filtered_force_n, dtype=np.float64),
        np.asarray(pad_gap_m, dtype=np.float64),
    )
    shape = parent.shape
    for name, values in (("previous_raw_force_n", state.previous_raw_force_n),
                         ("retract_latched", state.retract_latched),
                         ("safe_streak", state.safe_streak)):
        if np.asarray(values).shape != shape:
            raise ValueError(f"state {name} shape does not match inputs")
    finite = np.isfinite(parent) & np.isfinite(raw) & np.isfinite(filtered) & np.isfinite(gap)
    safe_parent = np.where(np.isfinite(parent), np.maximum(parent, 0.0), 0.0)

    if surface_kind == "flat":
        return {
            "force_command_n": safe_parent.copy(),
            "minimum_retract_velocity_m_s": np.zeros(shape),
            "predicted_force_n": np.where(finite, raw, cfg.hard_force_n),
            "raw_force_rate_n_s": np.zeros(shape),
            "intervention": np.zeros(shape, dtype=bool),
            "fault": ~finite,
            "flat_exact_parity": finite.copy(),
        }
    if surface_kind not in ("cylinder", "sphere", "freeform"):
        raise ValueError(f"unsupported surface kind: {surface_kind}")

    previous = np.asarray(state.previous_raw_force_n, dtype=np.float64)
    positive_delta = np.maximum(raw - previous, 0.0)
    predicted = raw + cfg.prediction_horizon_substeps * positive_delta
    predicted = np.where(finite, predicted, cfg.hard_force_n)
    trigger = (~finite) | (raw >= cfg.predictive_guard_n) | (
        predicted >= cfg.predictive_guard_n)
    latched = np.asarray(state.retract_latched, dtype=bool) | trigger
    # A triggering sample cannot simultaneously count toward release, even if
    # its current raw value is low and only its predicted rise crossed guard.
    release_sample = finite & ~trigger & (raw <= cfg.release_raw_force_n) & (
        filtered <= cfg.release_filtered_force_n)
    streak = np.where(latched & release_sample,
                      np.asarray(state.safe_streak, dtype=np.int64) + 1, 0)
    release = latched & (streak >= cfg.release_substeps)
    latched = latched & ~release
    streak = np.where(release, 0, streak)

    force_command = np.where(latched,
                             np.minimum(safe_parent, cfg.retract_force_floor_n),
                             safe_parent)
    force_command = np.minimum(force_command, safe_parent)
    retract = np.where(latched, cfg.retract_velocity_m_s, 0.0)
    state.previous_raw_force_n[...] = np.where(finite, raw, previous)
    state.retract_latched[...] = latched
    state.safe_streak[...] = streak
    return {
        "force_command_n": force_command,
        "minimum_retract_velocity_m_s": retract,
        "predicted_force_n": predicted,
        "raw_force_rate_n_s": positive_delta * 120.0,
        "intervention": latched,
        "fault": ~finite,
        "flat_exact_parity": np.zeros(shape, dtype=bool),
    }
