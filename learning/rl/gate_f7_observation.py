"""Isaac-free observation schema for the isolated Gate F7 ablation."""
from __future__ import annotations

import numpy as np

from learning.rl.gate_f_curved_geometry import SurfaceGeometrySpec, surface_height_normal


BASE14 = "base14"
NORMAL17 = "normal17"
NORMAL_CURVATURE20 = "normal_curvature20"
OBSERVATION_MODES = (BASE14, NORMAL17, NORMAL_CURVATURE20)

BASE_DIM = 14
NORMAL_FEATURE_NAMES = (
    "surface_normal_x",
    "surface_normal_y",
    "surface_normal_z",
)
CURVATURE_FEATURE_NAMES = (
    "surface_hessian_uu_norm",
    "surface_hessian_uv_norm",
    "surface_hessian_vv_norm",
)
# Ten inverse metres bounds the design surfaces while keeping R=0.60 m
# cylinder/sphere curvature away from numerical zero.
CURVATURE_SCALE_PER_M = 10.0


def observation_dim(mode: str) -> int:
    if mode == BASE14:
        return BASE_DIM
    if mode == NORMAL17:
        return BASE_DIM + 3
    if mode == NORMAL_CURVATURE20:
        return BASE_DIM + 6
    raise ValueError(f"unknown Gate F7 observation mode: {mode!r}")


def observation_feature_names(mode: str) -> tuple[str, ...]:
    from learning.rl.env.gate_d_observation import BASE_FEATURE_NAMES

    names = list(BASE_FEATURE_NAMES)
    if mode in (NORMAL17, NORMAL_CURVATURE20):
        names.extend(NORMAL_FEATURE_NAMES)
    elif mode != BASE14:
        raise ValueError(f"unknown Gate F7 observation mode: {mode!r}")
    if mode == NORMAL_CURVATURE20:
        names.extend(CURVATURE_FEATURE_NAMES)
    return tuple(names)


def local_geometry_features(
    spec: SurfaceGeometrySpec,
    u_m: float | np.ndarray,
    v_m: float | np.ndarray,
    *,
    finite_difference_m: float = 1.0e-3,
) -> np.ndarray:
    """Return normal XYZ plus normalized graph Hessian h_uu,h_uv,h_vv."""
    spec.validate()
    step = float(finite_difference_m)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("finite_difference_m must be finite and positive")
    u, v = np.broadcast_arrays(
        np.asarray(u_m, dtype=np.float64), np.asarray(v_m, dtype=np.float64))
    h, normal = surface_height_normal(
        spec.kind, spec.curvature_radius_m, spec.patch_size_m, u, v,
        freeform_seed=spec.freeform_seed)
    hp_u, _ = surface_height_normal(
        spec.kind, spec.curvature_radius_m, spec.patch_size_m, u + step, v,
        freeform_seed=spec.freeform_seed)
    hm_u, _ = surface_height_normal(
        spec.kind, spec.curvature_radius_m, spec.patch_size_m, u - step, v,
        freeform_seed=spec.freeform_seed)
    hp_v, _ = surface_height_normal(
        spec.kind, spec.curvature_radius_m, spec.patch_size_m, u, v + step,
        freeform_seed=spec.freeform_seed)
    hm_v, _ = surface_height_normal(
        spec.kind, spec.curvature_radius_m, spec.patch_size_m, u, v - step,
        freeform_seed=spec.freeform_seed)
    hpp, _ = surface_height_normal(
        spec.kind, spec.curvature_radius_m, spec.patch_size_m, u + step, v + step,
        freeform_seed=spec.freeform_seed)
    hpm, _ = surface_height_normal(
        spec.kind, spec.curvature_radius_m, spec.patch_size_m, u + step, v - step,
        freeform_seed=spec.freeform_seed)
    hmp, _ = surface_height_normal(
        spec.kind, spec.curvature_radius_m, spec.patch_size_m, u - step, v + step,
        freeform_seed=spec.freeform_seed)
    hmm, _ = surface_height_normal(
        spec.kind, spec.curvature_radius_m, spec.patch_size_m, u - step, v - step,
        freeform_seed=spec.freeform_seed)
    h = np.asarray(h, dtype=np.float64)
    huu = (np.asarray(hp_u) - 2.0 * h + np.asarray(hm_u)) / step**2
    hvv = (np.asarray(hp_v) - 2.0 * h + np.asarray(hm_v)) / step**2
    huv = (np.asarray(hpp) - np.asarray(hpm) - np.asarray(hmp) + np.asarray(hmm)) / (4.0 * step**2)
    curvature = np.stack((huu, huv, hvv), axis=-1) / CURVATURE_SCALE_PER_M
    curvature = np.clip(curvature, -1.0, 1.0)
    out = np.concatenate((np.asarray(normal, dtype=np.float64), curvature), axis=-1)
    if out.shape[-1] != 6 or not np.isfinite(out).all():
        raise RuntimeError("invalid Gate F7 local geometry features")
    return out.astype(np.float32)


def compose_observation(base14: np.ndarray, geometry6: np.ndarray, mode: str) -> np.ndarray:
    base = np.asarray(base14, dtype=np.float32)
    geometry = np.asarray(geometry6, dtype=np.float32)
    if base.shape[-1] != BASE_DIM or geometry.shape[-1] != 6:
        raise ValueError(f"expected base ...x14 and geometry ...x6, got {base.shape}, {geometry.shape}")
    if base.shape[:-1] != geometry.shape[:-1]:
        raise ValueError("base and geometry batch shapes must match")
    if mode == BASE14:
        out = base.copy()
    elif mode == NORMAL17:
        out = np.concatenate((base, geometry[..., :3]), axis=-1)
    elif mode == NORMAL_CURVATURE20:
        out = np.concatenate((base, geometry), axis=-1)
    else:
        raise ValueError(f"unknown Gate F7 observation mode: {mode!r}")
    if out.shape[-1] != observation_dim(mode) or not np.isfinite(out).all():
        raise RuntimeError("invalid composed Gate F7 observation")
    return out
