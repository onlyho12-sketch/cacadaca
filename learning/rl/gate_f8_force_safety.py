"""Pure base14 force-action shields for the isolated Gate F8 pilot."""
from __future__ import annotations

import numpy as np


CONTROL = "control"
STATIC_CAP = "static_cap"
PREDICTIVE_SHIELD = "predictive_shield"
SHIELD_MODES = (CONTROL, STATIC_CAP, PREDICTIVE_SHIELD)
STATIC_FORCE_ACTION_CAP = 0.50
PREDICTION_HORIZON_STEPS = 2.0
PREDICTIVE_START_N = 9.0
PREDICTIVE_ZERO_ACTION_N = 11.0
PREDICTIVE_MIN_ACTION = -0.50


def apply_force_action_shield(
    actions: np.ndarray,
    base14: np.ndarray,
    mode: str,
    *,
    curved: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return shielded actions and the per-row force-action cap.

    Base14 index 0 is force/10 and index 2 is force delta/5.  Only the force
    action is constrained; feed remains byte-for-byte unchanged.
    """
    values = np.asarray(actions, dtype=np.float32)
    obs = np.asarray(base14, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"actions must have shape (N,2), got {values.shape}")
    if obs.shape != (len(values), 14):
        raise ValueError(f"base14 must have shape {(len(values), 14)}, got {obs.shape}")
    if mode not in SHIELD_MODES:
        raise ValueError(f"mode must be one of {SHIELD_MODES}")
    if not np.isfinite(values).all() or not np.isfinite(obs).all():
        raise ValueError("actions/base14 contain NaN or Inf")
    out = np.clip(values, -1.0, 1.0).copy()
    cap = np.ones(len(values), dtype=np.float32)
    if mode == CONTROL or not curved:
        return out, cap
    cap.fill(STATIC_FORCE_ACTION_CAP)
    if mode == PREDICTIVE_SHIELD:
        force_n = obs[:, 0] * 10.0
        delta_n = obs[:, 2] * 5.0
        predicted_n = force_n + PREDICTION_HORIZON_STEPS * np.maximum(delta_n, 0.0)
        dynamic = 1.0 - (
            (predicted_n - PREDICTIVE_START_N)
            / (PREDICTIVE_ZERO_ACTION_N - PREDICTIVE_START_N)
        )
        dynamic = np.clip(dynamic, PREDICTIVE_MIN_ACTION, 1.0)
        cap = np.minimum(cap, dynamic.astype(np.float32))
    out[:, 0] = np.minimum(out[:, 0], cap)
    return out, cap
