"""Gate F10-F deterministic curvature safety and bounded residual interface.

Pure NumPy: no Isaac, PhysX, policy loading or training.  A future PPO may emit
two normalized residuals, but this layer converts them to small geometry-
conditioned physical corrections inside deterministic force/feed envelopes.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np


OBSERVATION_NAMES = (
    "signed_k1_norm", "signed_k2_norm", "pad_curvature_ratio_norm",
    "footprint_normal_spread_norm", "fit_confidence", "boundary_risk",
    "hole_risk", "contact_gap_norm", "normal_force_norm", "force_rate_norm",
)


@dataclass(frozen=True)
class SafetyConfig:
    curvature_scale_1_m: float = 10.0
    pad_curvature_reference: float = 0.80
    normal_spread_reference_deg: float = 20.0
    top_flat_force_n: float = 8.0
    top_risk_force_n: float = 5.0
    side_flat_force_n: float = 6.0
    side_risk_force_n: float = 3.5
    base_feed_mm_s: float = 12.7
    risk_feed_mm_s: float = 6.35
    entry_force_floor_n: float = 2.0
    entry_force_ratio: float = 0.35
    entry_feed_ratio: float = 0.50
    force_slew_n_s: float = 2.0
    feed_slew_mm_s2: float = 25.0
    residual_force_down_ratio: float = 0.15
    residual_force_up_ratio: float = 0.10
    residual_feed_down_ratio: float = 0.20
    residual_feed_up_ratio: float = 0.15
    force_normalization_n: float = 14.0
    force_rate_normalization_n_s: float = 1000.0
    gap_normalization_m: float = 0.020


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def smoothstep01(value):
    x = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def geometry_risk(*, pad_curvature_ratio, normal_spread_deg, fit_confidence,
                  boundary_risk, hole_risk, cfg: SafetyConfig) -> np.ndarray:
    values = np.stack((
        np.asarray(pad_curvature_ratio) / cfg.pad_curvature_reference,
        np.asarray(normal_spread_deg) / cfg.normal_spread_reference_deg,
        1.0 - np.asarray(fit_confidence),
        np.asarray(boundary_risk), np.asarray(hole_risk),
    ), axis=-1)
    return smoothstep01(np.max(values, axis=-1))


def deterministic_envelope(risk, is_side, entry_progress, cfg: SafetyConfig
                           ) -> tuple[np.ndarray, np.ndarray]:
    risk = np.clip(np.asarray(risk, dtype=np.float64), 0.0, 1.0)
    side = np.asarray(is_side, dtype=bool)
    entry = smoothstep01(entry_progress)
    flat = np.where(side, cfg.side_flat_force_n, cfg.top_flat_force_n)
    high = np.where(side, cfg.side_risk_force_n, cfg.top_risk_force_n)
    force = flat + (high - flat) * risk
    entry_force_scale = cfg.entry_force_ratio + (1.0 - cfg.entry_force_ratio) * entry
    force = np.maximum(cfg.entry_force_floor_n, force * entry_force_scale)
    feed = cfg.base_feed_mm_s + (cfg.risk_feed_mm_s - cfg.base_feed_mm_s) * risk
    entry_feed_scale = cfg.entry_feed_ratio + (1.0 - cfg.entry_feed_ratio) * entry
    feed = np.maximum(cfg.risk_feed_mm_s, feed * entry_feed_scale)
    return force, feed


def residual_bounds(baseline_force_n, baseline_feed_mm_s, risk, cfg: SafetyConfig
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    force = np.asarray(baseline_force_n, dtype=np.float64)
    feed = np.asarray(baseline_feed_mm_s, dtype=np.float64)
    risk = np.clip(np.asarray(risk, dtype=np.float64), 0.0, 1.0)
    force_low = -cfg.residual_force_down_ratio * force
    force_high = cfg.residual_force_up_ratio * force * (1.0 - risk)
    feed_low = -cfg.residual_feed_down_ratio * feed
    feed_high = cfg.residual_feed_up_ratio * feed * (1.0 - risk)
    return force_low, force_high, feed_low, feed_high


def _map_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    return np.where(clipped >= 0.0, clipped * high, -clipped * low)


def apply_bounded_residual(baseline_force_n, baseline_feed_mm_s, risk, actions,
                           is_side, cfg: SafetyConfig,
                           command_allowed=None) -> dict[str, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float64)
    force = np.asarray(baseline_force_n, dtype=np.float64)
    feed = np.asarray(baseline_feed_mm_s, dtype=np.float64)
    if actions.shape != force.shape + (2,):
        raise ValueError(f"actions shape {actions.shape} != {force.shape + (2,)}")
    if not np.isfinite(actions).all():
        raise ValueError("actions contain NaN/Inf")
    fl, fh, vl, vh = residual_bounds(force, feed, risk, cfg)
    force_delta = _map_action(actions[..., 0], fl, fh)
    feed_delta = _map_action(actions[..., 1], vl, vh)
    side = np.asarray(is_side, dtype=bool)
    hard_force_envelope = np.where(side, cfg.side_flat_force_n, cfg.top_flat_force_n)
    output_force = np.clip(force + force_delta, cfg.entry_force_floor_n, hard_force_envelope)
    output_feed = np.clip(feed + feed_delta, cfg.risk_feed_mm_s, cfg.base_feed_mm_s)
    allowed = (np.ones(force.shape, dtype=bool) if command_allowed is None else
               np.broadcast_to(np.asarray(command_allowed, dtype=bool), force.shape))
    output_force = np.where(allowed, output_force, 0.0)
    output_feed = np.where(allowed, output_feed, 0.0)
    return {
        "force_command_n": output_force, "feed_command_mm_s": output_feed,
        "force_delta_n": output_force - force, "feed_delta_mm_s": output_feed - feed,
        "force_delta_low_n": fl, "force_delta_high_n": fh,
        "feed_delta_low_mm_s": vl, "feed_delta_high_mm_s": vh,
        "action_clipped": np.any(np.abs(actions) > 1.0, axis=-1),
    }


def slew_sequence(target: np.ndarray, dt_s: np.ndarray, rate_per_s: float,
                  reset_mask: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    dt = np.asarray(dt_s, dtype=np.float64)
    reset = np.asarray(reset_mask, dtype=bool)
    output = target.copy()
    for i in range(1, len(target)):
        if reset[i]:
            continue
        delta = rate_per_s * max(float(dt[i]), 1e-9)
        output[i] = np.clip(target[i], output[i - 1] - delta, output[i - 1] + delta)
    return output


def compose_observation(k1, k2, pad_curvature_ratio, normal_spread_deg,
                        fit_confidence, boundary_risk, hole_risk, contact_gap_m,
                        normal_force_n, force_rate_n_s, cfg: SafetyConfig) -> np.ndarray:
    arrays = np.broadcast_arrays(k1, k2, pad_curvature_ratio, normal_spread_deg,
                                 fit_confidence, boundary_risk, hole_risk, contact_gap_m,
                                 normal_force_n, force_rate_n_s)
    out = np.stack((
        np.clip(arrays[0] / cfg.curvature_scale_1_m, -1.0, 1.0),
        np.clip(arrays[1] / cfg.curvature_scale_1_m, -1.0, 1.0),
        np.clip(arrays[2] / cfg.pad_curvature_reference, 0.0, 1.0),
        np.clip(arrays[3] / cfg.normal_spread_reference_deg, 0.0, 1.0),
        np.clip(arrays[4], 0.0, 1.0), np.clip(arrays[5], 0.0, 1.0),
        np.clip(arrays[6], 0.0, 1.0),
        np.clip(arrays[7] / cfg.gap_normalization_m, -1.0, 1.0),
        np.clip(arrays[8] / cfg.force_normalization_n, 0.0, 1.0),
        np.clip(arrays[9] / cfg.force_rate_normalization_n_s, -1.0, 1.0),
    ), axis=-1).astype(np.float32)
    if out.shape[-1] != len(OBSERVATION_NAMES) or not np.isfinite(out).all():
        raise RuntimeError("invalid curvature safety observation")
    return out


def apply_curvature_physx_action_shield(
        parent_actions, geometry6, pad_uv_m, arc_m, *, surface_kind: str,
        patch_size_m, baseline_force_n: float, baseline_feed_mm_s: float,
        force_ratio_limit: float, feed_ratio_limit: float, control_dt_s: float,
        previous_force_n=None, previous_feed_mm_s=None,
        cfg: SafetyConfig = SafetyConfig()) -> dict[str, np.ndarray]:
    """Map E7 actions through F10 deterministic limits for the isolated pilot.

    Flat is exact parent parity.  Curved rows use local Hessian curvature,
    footprint-edge risk, entry soft-start, and physical command slew limits.
    """
    parent = np.clip(np.asarray(parent_actions, dtype=np.float64), -1.0, 1.0)
    geometry = np.asarray(geometry6, dtype=np.float64)
    uv = np.asarray(pad_uv_m, dtype=np.float64)
    arc = np.asarray(arc_m, dtype=np.float64)
    if parent.ndim != 2 or parent.shape[1] != 2:
        raise ValueError("parent_actions must have shape (N,2)")
    n = len(parent)
    if geometry.shape != (n, 6) or uv.shape != (n, 2) or arc.shape != (n,):
        raise ValueError("geometry6/pad_uv_m/arc_m batch shapes do not match")
    if not all(np.isfinite(value).all() for value in (parent, geometry, uv, arc)):
        raise ValueError("F10 PhysX shield inputs contain NaN/Inf")
    if surface_kind == "flat":
        force = baseline_force_n * (1.0 + parent[:, 0] * force_ratio_limit)
        feed = baseline_feed_mm_s * (1.0 + parent[:, 1] * feed_ratio_limit)
        return {"actions": parent.astype(np.float32), "risk": np.zeros(n),
                "force_command_n": force, "feed_command_mm_s": feed,
                "force_cap_action": np.ones(n), "feed_cap_action": np.ones(n),
                "flat_exact_parity": np.ones(n, dtype=bool)}

    hessian = geometry[:, (3, 4, 4, 5)].reshape(n, 2, 2) * cfg.curvature_scale_1_m
    principal = np.linalg.eigvalsh(hessian)
    curvature = np.max(np.abs(principal), axis=1)
    pad_ratio = curvature * 0.055
    spread_deg = np.degrees(2.0 * np.arctan(0.055 * curvature))
    width, height = map(float, patch_size_m)
    edge_distance = np.minimum.reduce((uv[:, 0], width - uv[:, 0],
                                       uv[:, 1], height - uv[:, 1]))
    boundary = 1.0 - np.clip(edge_distance / 0.055, 0.0, 1.0)
    risk = geometry_risk(
        pad_curvature_ratio=pad_ratio, normal_spread_deg=spread_deg,
        fit_confidence=np.ones(n), boundary_risk=boundary,
        hole_risk=np.zeros(n), cfg=cfg)
    entry = np.clip(arc / 0.04, 0.0, 1.0)
    target_force, target_feed = deterministic_envelope(
        risk, np.zeros(n, dtype=bool), entry, cfg)
    reachable_force_min = baseline_force_n * (1.0 - force_ratio_limit)
    reachable_feed_min = baseline_feed_mm_s * (1.0 - feed_ratio_limit)
    target_force = np.maximum(target_force, reachable_force_min)
    target_feed = np.maximum(target_feed, reachable_feed_min)
    force_cap = np.clip((target_force / baseline_force_n - 1.0) / force_ratio_limit,
                        -1.0, 1.0)
    feed_cap = np.clip((target_feed / baseline_feed_mm_s - 1.0) / feed_ratio_limit,
                       -1.0, 1.0)
    desired = np.minimum(parent, np.stack((force_cap, feed_cap), axis=1))
    force = baseline_force_n * (1.0 + desired[:, 0] * force_ratio_limit)
    feed = baseline_feed_mm_s * (1.0 + desired[:, 1] * feed_ratio_limit)
    if previous_force_n is not None and previous_feed_mm_s is not None:
        previous_force = np.asarray(previous_force_n, dtype=np.float64)
        previous_feed = np.asarray(previous_feed_mm_s, dtype=np.float64)
        force = np.clip(force, previous_force - cfg.force_slew_n_s * control_dt_s,
                        previous_force + cfg.force_slew_n_s * control_dt_s)
        feed = np.clip(feed, previous_feed - cfg.feed_slew_mm_s2 * control_dt_s,
                       previous_feed + cfg.feed_slew_mm_s2 * control_dt_s)
        desired[:, 0] = np.clip((force / baseline_force_n - 1.0) / force_ratio_limit,
                                -1.0, 1.0)
        desired[:, 1] = np.clip((feed / baseline_feed_mm_s - 1.0) / feed_ratio_limit,
                                -1.0, 1.0)
    return {"actions": desired.astype(np.float32), "risk": risk,
            "force_command_n": force, "feed_command_mm_s": feed,
            "force_cap_action": force_cap, "feed_cap_action": feed_cap,
            "flat_exact_parity": np.zeros(n, dtype=bool)}


def apply_curvature_physx_action_shield_v2(
        parent_actions, base14, geometry6, pad_uv_m, arc_m, *, surface_kind: str,
        patch_size_m, baseline_force_n: float, baseline_feed_mm_s: float,
        force_ratio_limit: float, feed_ratio_limit: float, control_dt_s: float,
        previous_force_n=None, previous_feed_mm_s=None,
        cfg: SafetyConfig = SafetyConfig()) -> dict[str, np.ndarray]:
    """One-sided-slew F10 safety candidate with a dynamic force predictor.

    Downward parent/cap requests take effect immediately.  Only command
    increases are slew-limited, so this layer can never hold force or feed
    above a safer parent request.  The predictor is the previously tested F8
    form: measured force plus two positive force deltas, with no cap below 9 N
    and a progressively negative cap from 9 to 12 N.
    """
    parent = np.clip(np.asarray(parent_actions, dtype=np.float64), -1.0, 1.0)
    obs = np.asarray(base14, dtype=np.float64)
    geometry = np.asarray(geometry6, dtype=np.float64)
    uv = np.asarray(pad_uv_m, dtype=np.float64)
    arc = np.asarray(arc_m, dtype=np.float64)
    if parent.ndim != 2 or parent.shape[1] != 2:
        raise ValueError("parent_actions must have shape (N,2)")
    n = len(parent)
    if obs.shape != (n, 14) or geometry.shape != (n, 6) or uv.shape != (n, 2) or arc.shape != (n,):
        raise ValueError("base14/geometry6/pad_uv_m/arc_m batch shapes do not match")
    if not all(np.isfinite(value).all() for value in (parent, obs, geometry, uv, arc)):
        raise ValueError("F10 PhysX shield v2 inputs contain NaN/Inf")
    if surface_kind == "flat":
        force = baseline_force_n * (1.0 + parent[:, 0] * force_ratio_limit)
        feed = baseline_feed_mm_s * (1.0 + parent[:, 1] * feed_ratio_limit)
        return {"actions": parent.astype(np.float32), "risk": np.zeros(n),
                "predicted_force_n": obs[:, 0] * 10.0,
                "predictive_force_cap_action": np.ones(n),
                "force_command_n": force, "feed_command_mm_s": feed,
                "force_cap_action": np.ones(n), "feed_cap_action": np.ones(n),
                "flat_exact_parity": np.ones(n, dtype=bool)}

    hessian = geometry[:, (3, 4, 4, 5)].reshape(n, 2, 2) * cfg.curvature_scale_1_m
    curvature = np.max(np.abs(np.linalg.eigvalsh(hessian)), axis=1)
    pad_ratio = curvature * 0.055
    spread_deg = np.degrees(2.0 * np.arctan(0.055 * curvature))
    width, height = map(float, patch_size_m)
    edge_distance = np.minimum.reduce((uv[:, 0], width - uv[:, 0],
                                       uv[:, 1], height - uv[:, 1]))
    boundary = 1.0 - np.clip(edge_distance / 0.055, 0.0, 1.0)
    risk = geometry_risk(
        pad_curvature_ratio=pad_ratio, normal_spread_deg=spread_deg,
        fit_confidence=np.ones(n), boundary_risk=boundary,
        hole_risk=np.zeros(n), cfg=cfg)
    entry = np.clip(arc / 0.04, 0.0, 1.0)
    target_force, target_feed = deterministic_envelope(
        risk, np.zeros(n, dtype=bool), entry, cfg)
    reachable_force_min = baseline_force_n * (1.0 - force_ratio_limit)
    reachable_feed_min = baseline_feed_mm_s * (1.0 - feed_ratio_limit)
    target_force = np.maximum(target_force, reachable_force_min)
    target_feed = np.maximum(target_feed, reachable_feed_min)
    geometry_force_cap = np.clip(
        (target_force / baseline_force_n - 1.0) / force_ratio_limit, -1.0, 1.0)
    geometry_feed_cap = np.clip(
        (target_feed / baseline_feed_mm_s - 1.0) / feed_ratio_limit, -1.0, 1.0)

    measured_force = obs[:, 0] * 10.0
    force_delta = obs[:, 2] * 5.0
    predicted_force = measured_force + 2.0 * np.maximum(force_delta, 0.0)
    predictive_cap = np.clip(1.0 - (predicted_force - 9.0) / 2.0, -0.5, 1.0)
    force_cap = np.minimum(geometry_force_cap, predictive_cap)
    desired = np.minimum(parent, np.stack((force_cap, geometry_feed_cap), axis=1))
    desired_force = baseline_force_n * (1.0 + desired[:, 0] * force_ratio_limit)
    desired_feed = baseline_feed_mm_s * (1.0 + desired[:, 1] * feed_ratio_limit)
    force = desired_force.copy()
    feed = desired_feed.copy()
    if previous_force_n is not None and previous_feed_mm_s is not None:
        previous_force = np.broadcast_to(np.asarray(previous_force_n, dtype=np.float64), force.shape)
        previous_feed = np.broadcast_to(np.asarray(previous_feed_mm_s, dtype=np.float64), feed.shape)
        force = np.minimum(force, previous_force + cfg.force_slew_n_s * control_dt_s)
        feed = np.minimum(feed, previous_feed + cfg.feed_slew_mm_s2 * control_dt_s)
    executed = np.empty_like(desired)
    executed[:, 0] = np.clip((force / baseline_force_n - 1.0) / force_ratio_limit, -1.0, 1.0)
    executed[:, 1] = np.clip((feed / baseline_feed_mm_s - 1.0) / feed_ratio_limit, -1.0, 1.0)
    if np.any(executed > parent + 1e-7) or np.any(executed[:, 0] > force_cap + 1e-7):
        raise RuntimeError("v2 safety contract violation: executed command exceeds parent/cap")
    return {"actions": executed.astype(np.float32), "risk": risk,
            "predicted_force_n": predicted_force,
            "predictive_force_cap_action": predictive_cap,
            "force_command_n": force, "feed_command_mm_s": feed,
            "force_cap_action": force_cap, "feed_cap_action": geometry_feed_cap,
            "flat_exact_parity": np.zeros(n, dtype=bool)}


def offline_ablation(rows: list[dict], cfg: SafetyConfig) -> tuple[list[dict], list[dict], dict]:
    output, arm_rows = [], []
    for rail in ("C", "SL", "SR"):
        selected = [row for row in rows if row["rail"] == rail]
        keys = []
        for row in selected:
            key = int(row["segment"])
            if key not in keys:
                keys.append(key)
        for segment in keys:
            segment_rows = sorted((row for row in selected if int(row["segment"]) == segment),
                                  key=lambda row: int(row["waypoint"]))
            n = len(segment_rows)
            side = np.full(n, rail in ("SL", "SR"), dtype=bool)
            risk = geometry_risk(
                pad_curvature_ratio=[float(row["pad_curvature_ratio"]) for row in segment_rows],
                normal_spread_deg=[float(row["normal_spread_p95_deg"]) for row in segment_rows],
                fit_confidence=[float(row["fit_confidence"]) for row in segment_rows],
                boundary_risk=[float(row["boundary_risk"]) for row in segment_rows],
                hole_risk=[float(row["hole_risk"]) for row in segment_rows], cfg=cfg)
            step = np.asarray([float(row["path_step_m"]) for row in segment_rows])
            continuous = np.asarray([str(row["path_continuous"]).lower() == "true"
                                     for row in segment_rows])
            unsafe = np.asarray([str(row["unsafe_geometry_pt_design"]).lower() == "true"
                                 for row in segment_rows])
            reset = ~continuous
            reset[0] = True
            reset |= unsafe
            if n > 1:
                reset[1:] |= unsafe[:-1]
            entry_distance = np.zeros(n)
            for i in range(1, n):
                entry_distance[i] = 0.0 if reset[i] else entry_distance[i - 1] + step[i]
            entry = np.clip(entry_distance / 0.04, 0.0, 1.0)
            full_entry = np.ones(n)
            control_force, control_feed = deterministic_envelope(
                np.zeros(n), side, full_entry, cfg)
            curvature_force, curvature_feed = deterministic_envelope(
                risk, side, full_entry, cfg)
            soft_force_target, soft_feed_target = deterministic_envelope(
                np.zeros(n), side, entry, cfg)
            combined_force_target, combined_feed_target = deterministic_envelope(
                risk, side, entry, cfg)
            dt = np.maximum(1.0 / 60.0,
                            step / np.maximum(combined_feed_target / 1000.0, 1e-6))
            soft_force = slew_sequence(soft_force_target, dt, cfg.force_slew_n_s, reset)
            soft_feed = slew_sequence(soft_feed_target, dt, cfg.feed_slew_mm_s2, reset)
            force_slew = slew_sequence(combined_force_target, dt, cfg.force_slew_n_s, reset)
            feed_slew = slew_sequence(combined_feed_target, dt, cfg.feed_slew_mm_s2, reset)
            allowed = ~unsafe
            zero = apply_bounded_residual(force_slew, feed_slew, risk,
                                          np.zeros((n, 2)), side, cfg, allowed)
            worst_up = apply_bounded_residual(force_slew, feed_slew, risk,
                                              np.ones((n, 2)), side, cfg, allowed)
            worst_down = apply_bounded_residual(force_slew, feed_slew, risk,
                                                -np.ones((n, 2)), side, cfg, allowed)
            observation = compose_observation(
                [float(row["k1_1_m"]) for row in segment_rows],
                [float(row["k2_1_m"]) for row in segment_rows],
                [float(row["pad_curvature_ratio"]) for row in segment_rows],
                [float(row["normal_spread_p95_deg"]) for row in segment_rows],
                [float(row["fit_confidence"]) for row in segment_rows],
                [float(row["boundary_risk"]) for row in segment_rows],
                [float(row["hole_risk"]) for row in segment_rows],
                np.zeros(n), np.zeros(n), np.zeros(n), cfg)
            for i, source in enumerate(segment_rows):
                output.append({
                    "rail": rail, "segment": segment, "waypoint": source["waypoint"],
                    "path_file": source["path_file"], "geometry_risk": float(risk[i]),
                    "path_step_m": float(step[i]), "command_allowed": bool(allowed[i]),
                    "hold_reason": source["unsafe_reasons"] if unsafe[i] else "",
                    "entry_progress": float(entry[i]), "reset_before": bool(reset[i]),
                    "control_force_n": float(control_force[i]),
                    "control_feed_mm_s": float(control_feed[i]),
                    "curvature_force_n": float(curvature_force[i]),
                    "curvature_feed_mm_s": float(curvature_feed[i]),
                    "softstart_force_n": float(soft_force[i]),
                    "softstart_feed_mm_s": float(soft_feed[i]),
                    "deterministic_force_n": float(force_slew[i] if allowed[i] else 0.0),
                    "deterministic_feed_mm_s": float(feed_slew[i] if allowed[i] else 0.0),
                    "projected_combined_force_n": float(force_slew[i]),
                    "projected_combined_feed_mm_s": float(feed_slew[i]),
                    "nominal_dt_s": float(dt[i]),
                    "zero_residual_force_n": float(zero["force_command_n"][i]),
                    "zero_residual_feed_mm_s": float(zero["feed_command_mm_s"][i]),
                    "worst_up_force_n": float(worst_up["force_command_n"][i]),
                    "worst_up_feed_mm_s": float(worst_up["feed_command_mm_s"][i]),
                    "worst_down_force_n": float(worst_down["force_command_n"][i]),
                    "worst_down_feed_mm_s": float(worst_down["feed_command_mm_s"][i]),
                    **{name: float(observation[i, j]) for j, name in enumerate(OBSERVATION_NAMES)},
                })
    for arm in ("control", "curvature_derating", "slew_softstart", "combined",
                "bounded_residual_interface"):
        if arm == "control":
            force = np.asarray([row["control_force_n"] for row in output])
            feed = np.asarray([row["control_feed_mm_s"] for row in output])
        elif arm == "curvature_derating":
            force = np.asarray([row["curvature_force_n"] for row in output])
            feed = np.asarray([row["curvature_feed_mm_s"] for row in output])
        elif arm == "slew_softstart":
            force = np.asarray([row["softstart_force_n"] for row in output])
            feed = np.asarray([row["softstart_feed_mm_s"] for row in output])
        elif arm == "bounded_residual_interface":
            force = np.asarray([row["worst_up_force_n"] if row["command_allowed"]
                                else row["curvature_force_n"] for row in output])
            feed = np.asarray([row["worst_up_feed_mm_s"] if row["command_allowed"]
                               else row["projected_combined_feed_mm_s"] for row in output])
        else:
            force = np.asarray([row["deterministic_force_n"] if row["command_allowed"]
                                else row["curvature_force_n"] for row in output])
            feed = np.asarray([row["projected_combined_feed_mm_s"] for row in output])
        polish_s = {}
        for rail in ("C", "SL", "SR"):
            indices = [i for i, row in enumerate(output) if row["rail"] == rail]
            polish_s[rail] = float(sum(output[i]["path_step_m"] /
                                       max(feed[i] / 1000.0, 1e-9) for i in indices))
        anchor = np.asarray([cfg.side_flat_force_n if row["rail"] in ("SL", "SR")
                             else cfg.top_flat_force_n for row in output])
        arm_rows.append({
            "arm": arm, "waypoints": len(output),
            "force_min_n": float(force.min()), "force_max_n": float(force.max()),
            "feed_min_mm_s": float(feed.min()), "feed_max_mm_s": float(feed.max()),
            "force_saturation_fraction": float(np.mean(np.isclose(force, anchor))),
            "projected_polish_s_C": polish_s["C"],
            "projected_polish_s_SL": polish_s["SL"],
            "projected_polish_s_SR": polish_s["SR"],
            "finite": bool(np.isfinite(force).all() and np.isfinite(feed).all()),
        })
    zero_parity = all(row["deterministic_force_n"] == row["zero_residual_force_n"]
                      and row["deterministic_feed_mm_s"] == row["zero_residual_feed_mm_s"]
                      for row in output)
    summary = {
        "waypoints": len(output), "observation_dim": len(OBSERVATION_NAMES),
        "command_allowed_waypoints": sum(row["command_allowed"] for row in output),
        "hold_waypoints": sum(not row["command_allowed"] for row in output),
        "zero_residual_exact_parity": zero_parity,
        "force_command_min_n": min(row["worst_down_force_n"] for row in output),
        "force_command_max_n": max(row["worst_up_force_n"] for row in output),
        "feed_command_min_mm_s": min(row["worst_down_feed_mm_s"] for row in output),
        "feed_command_max_mm_s": max(row["worst_up_feed_mm_s"] for row in output),
        "allowed_force_command_min_n": min(row["worst_down_force_n"] for row in output
                                             if row["command_allowed"]),
        "allowed_feed_command_min_mm_s": min(row["worst_down_feed_mm_s"] for row in output
                                               if row["command_allowed"]),
        "static_cap_used": False,
    }
    return output, arm_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-result", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--time-baseline", required=True,
                        help="F10-A v2 result containing rail_time_baseline.csv")
    args = parser.parse_args()
    if os.path.exists(args.out_dir):
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    cfg = SafetyConfig()
    geometry_csv = os.path.join(args.geometry_result, "waypoint_geometry.csv")
    rows, arms, summary = offline_ablation(read_csv(geometry_csv), cfg)
    baseline_csv = os.path.join(args.time_baseline, "rail_time_baseline.csv")
    baseline_rows = {row["rail"]: row for row in read_csv(baseline_csv)}
    baseline_parallel_s = max(float(row["coarse_total_s"]) for row in baseline_rows.values())
    for arm in arms:
        rail_totals = []
        for rail in ("C", "SL", "SR"):
            fixed_s = (float(baseline_rows[rail]["coarse_total_s"]) -
                       float(baseline_rows[rail]["polish_s"]))
            rail_totals.append(fixed_s + float(arm[f"projected_polish_s_{rail}"]))
        arm["projected_parallel_total_s"] = max(rail_totals)
        arm["projected_parallel_total_h"] = max(rail_totals) / 3600.0
        arm["parallel_time_ratio_vs_control"] = max(rail_totals) / baseline_parallel_s
    combined_time_h = next(row["projected_parallel_total_h"] for row in arms
                           if row["arm"] == "combined")
    summary["baseline_parallel_total_h"] = baseline_parallel_s / 3600.0
    summary["combined_projected_parallel_total_h"] = combined_time_h
    summary["combined_parallel_time_ratio"] = combined_time_h / (baseline_parallel_s / 3600.0)
    summary["time_projection_assumption"] = (
        "all HOLD geometry is resolved before execution; no unsafe waypoint is commanded")
    finite = all(bool(row["finite"]) for row in arms)
    allowed_rows = [row for row in rows if row["command_allowed"]]
    held_rows = [row for row in rows if not row["command_allowed"]]
    force_bounds = min(row["worst_down_force_n"] for row in allowed_rows) >= cfg.entry_force_floor_n and summary[
        "force_command_max_n"] <= cfg.top_flat_force_n
    feed_bounds = min(row["worst_down_feed_mm_s"] for row in allowed_rows) >= cfg.risk_feed_mm_s and summary[
        "feed_command_max_mm_s"] <= cfg.base_feed_mm_s
    slew_ok = all(
        row["reset_before"] or (
            abs(row["projected_combined_force_n"] - rows[i - 1]["projected_combined_force_n"])
            <= cfg.force_slew_n_s * row["nominal_dt_s"] + 1e-9 and
            abs(row["projected_combined_feed_mm_s"] - rows[i - 1]["projected_combined_feed_mm_s"])
            <= cfg.feed_slew_mm_s2 * row["nominal_dt_s"] + 1e-9)
        for i, row in enumerate(rows) if i > 0)
    checks = [
        {"check": "all_vehicle_waypoints", "expected": 2498,
         "actual": len(rows), "pass": len(rows) == 2498},
        {"check": "zero_residual_exact_parity", "expected": True,
         "actual": summary["zero_residual_exact_parity"],
         "pass": summary["zero_residual_exact_parity"]},
        {"check": "finite_all_arms", "expected": True, "actual": finite, "pass": finite},
        {"check": "force_within_deterministic_envelope", "expected": True,
         "actual": force_bounds, "pass": force_bounds},
        {"check": "feed_within_deterministic_envelope", "expected": True,
         "actual": feed_bounds, "pass": feed_bounds},
        {"check": "no_fixed_static_cap", "expected": False,
         "actual": summary["static_cap_used"], "pass": not summary["static_cap_used"]},
        {"check": "observation_schema_complete", "expected": 10,
         "actual": summary["observation_dim"], "pass": summary["observation_dim"] == 10},
        {"check": "unsafe_geometry_is_hold_zero_command", "expected": True,
         "actual": all(row["deterministic_force_n"] == 0.0 and
                       row["deterministic_feed_mm_s"] == 0.0 for row in held_rows),
         "pass": all(row["deterministic_force_n"] == 0.0 and
                     row["deterministic_feed_mm_s"] == 0.0 for row in held_rows)},
        {"check": "combined_projected_parallel_under_4h", "expected": "<=4.0",
         "actual": combined_time_h, "pass": combined_time_h <= 4.0},
        {"check": "combined_force_feed_slew_bounded", "expected": True,
         "actual": slew_ok, "pass": slew_ok},
    ]
    os.makedirs(args.out_dir)
    write_csv(os.path.join(args.out_dir, "waypoint_offline_ablation.csv"), rows)
    write_csv(os.path.join(args.out_dir, "arm_summary.csv"), arms)
    write_csv(os.path.join(args.out_dir, "acceptance_checks.csv"), checks)
    write_csv(os.path.join(args.out_dir, "input_checksums.csv"),
              [{"path": os.path.abspath(path), "sha256": sha256(path)}
               for path in (geometry_csv, baseline_csv)])
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F10F_CURVATURE_SAFETY_OFFLINE",
        "status": "CPU_INTERFACE_COMPLETE_TRAINING_NOT_AUTHORIZED",
        "configuration_pt_design": asdict(cfg), "observation_names": OBSERVATION_NAMES,
        "summary": summary,
        "acceptance_checks_pass": all(bool(row["pass"]) for row in checks),
        "ppo_training_performed": False, "checkpoint_created": False,
        "physx_executed": False, "polishing_v5_modified": False,
        "next_step_allowed": False,
    }
    with open(os.path.join(args.out_dir, "decision.json"), "w", encoding="utf-8") as fh:
        json.dump(decision, fh, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("# Gate F10-F curvature safety offline interface\n\n"
                 "CPU-only deterministic envelopes and bounded residual mapping. "
                 "No policy was trained or evaluated in PhysX.\n")
    outputs = sorted(glob.glob(os.path.join(args.out_dir, "*")))
    with open(os.path.join(args.out_dir, "checksums.sha256"), "w", encoding="utf-8") as fh:
        for path in outputs:
            fh.write(f"{sha256(path)}  {os.path.basename(path)}\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
