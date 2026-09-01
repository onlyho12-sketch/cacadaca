"""Pure NumPy bounded residual composition for the isolated F10-H PPO."""
from __future__ import annotations

import numpy as np


def compose_bounded_residual_actions(
    parent_safe_actions,
    residual_actions,
    geometry_risk,
    force_cap_action,
    feed_cap_action,
    *,
    surface_kind: str,
    force_down: float = 0.15,
    force_up: float = 0.10,
    feed_down: float = 0.20,
    feed_up: float = 0.15,
) -> dict[str, np.ndarray]:
    """Compose small residuals inside deterministic action-space caps."""
    parent = np.asarray(parent_safe_actions, dtype=np.float64)
    residual = np.asarray(residual_actions, dtype=np.float64)
    risk = np.asarray(geometry_risk, dtype=np.float64)
    force_cap = np.asarray(force_cap_action, dtype=np.float64)
    feed_cap = np.asarray(feed_cap_action, dtype=np.float64)
    if parent.ndim != 2 or parent.shape[1] != 2 or residual.shape != parent.shape:
        raise ValueError("parent/residual actions must have matching (N,2) shape")
    n = len(parent)
    if risk.shape != (n,) or force_cap.shape != (n,) or feed_cap.shape != (n,):
        raise ValueError("risk/cap arrays must have shape (N,)")
    if not all(np.isfinite(value).all() for value in
               (parent, residual, risk, force_cap, feed_cap)):
        raise ValueError("bounded residual inputs contain NaN/Inf")
    parent = np.clip(parent, -1.0, 1.0)
    residual = np.clip(residual, -1.0, 1.0)
    risk = np.clip(risk, 0.0, 1.0)
    if surface_kind == "flat":
        return {
            "actions": parent.astype(np.float32),
            "residual_delta": np.zeros_like(parent, dtype=np.float32),
            "residual_suppressed": np.ones(n, dtype=bool),
            "cap_contract": np.ones(n, dtype=bool),
        }
    if surface_kind not in ("cylinder", "sphere", "freeform"):
        raise ValueError(f"unsupported surface kind: {surface_kind}")

    force_delta = np.where(
        residual[:, 0] >= 0.0,
        residual[:, 0] * force_up * (1.0 - risk),
        residual[:, 0] * force_down,
    )
    feed_delta = np.where(
        residual[:, 1] >= 0.0,
        residual[:, 1] * feed_up * (1.0 - risk),
        residual[:, 1] * feed_down,
    )
    desired = parent + np.stack((force_delta, feed_delta), axis=1)
    desired[:, 0] = np.minimum(desired[:, 0], np.clip(force_cap, -1.0, 1.0))
    desired[:, 1] = np.minimum(desired[:, 1], np.clip(feed_cap, -1.0, 1.0))
    desired = np.clip(desired, -1.0, 1.0)
    delta = desired - parent
    contract = ((desired[:, 0] <= force_cap + 1e-9)
                & (desired[:, 1] <= feed_cap + 1e-9))
    if not contract.all():
        raise RuntimeError("bounded residual escaped deterministic caps")
    return {
        "actions": desired.astype(np.float32),
        "residual_delta": delta.astype(np.float32),
        "residual_suppressed": np.zeros(n, dtype=bool),
        "cap_contract": contract,
    }
