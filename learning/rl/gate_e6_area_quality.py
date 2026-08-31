"""Gate E6 sub-tile quality-area diagnostics.

The established environment makes its episode decision from one ROI scalar and
reports a 5x5 tile diagnostic.  This module does not change that decision.  It
adds a PT-DESIGN, 10 mm local-window view so that a tile is not described as
fully polished merely because its aggregate values pass.

All maps have one value per existing 2 mm ROI surface cell.  Local Ra, Rz, and
GU use a centered 5x5-cell (10x10 mm) window.  The scratch criterion is the
established final rule applied cellwise: improved from the initial depth, or an
initial depth below 0.05 um.  Clearcoat and temperature remain separate safety
diagnostics and are never folded into the four-component quality area.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import maximum_filter, uniform_filter

from learning.polytwin import config as C
from learning.polytwin.gloss_proxy import (
    GlossProxyConfig,
    TARGET_GU,
    gu_from_relative,
)
from learning.polytwin.roughness_metrics import residual_scratch_depth_um


AREA_DIAGNOSTIC_VERSION = "gate_e6_local_window_area_v1"


@dataclass(frozen=True)
class AreaQualityTargets:
    """PT-DESIGN thresholds inherited from the established final rule."""

    gu_min: float = TARGET_GU
    ra_max_um: float = 0.20
    rz_max_um: float = 2.0
    scratch_initial_negligible_um: float = 0.05
    clearcoat_min_um: float = C.CLEARCOAT_SAFETY_LIMIT_UM
    temperature_max_c: float = 80.0
    local_window_cells: int = 5
    tiles_x: int = 5
    tiles_y: int = 5


def _validate(state, targets: AreaQualityTargets) -> None:
    shape = tuple(state.shape)
    if len(shape) != 2 or min(shape) < targets.local_window_cells:
        raise ValueError(f"surface shape {shape} is too small for local window")
    if targets.local_window_cells < 3 or targets.local_window_cells % 2 != 1:
        raise ValueError("local_window_cells must be an odd integer >= 3")
    if targets.tiles_x <= 0 or targets.tiles_y <= 0:
        raise ValueError("tile counts must be positive")
    if not np.isfinite(float(state.resolution_m)) or state.resolution_m <= 0:
        raise ValueError("surface resolution must be finite and positive")


def _window_stack(values: np.ndarray, size: int) -> np.ndarray:
    pad = size // 2
    padded = np.pad(np.asarray(values, dtype=np.float64), pad, mode="reflect")
    windows = sliding_window_view(padded, (size, size))
    return windows.reshape(values.shape[0], values.shape[1], size * size)


def _local_texture_maps(height_um: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return local plane-detrended Ra and five-high/five-low Rz maps.

    The established scalar Rz falls back to the five highest and five lowest
    samples whenever a patch has fewer than five independent extrema.  The
    5x5 local extension uses that fallback explicitly and is therefore tagged
    PT-DESIGN rather than presented as a physical profilometer measurement.
    """
    windows = _window_stack(height_um, size)
    axis = np.arange(size, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    design = np.stack(
        [np.ones(size * size), xx.ravel(), yy.ravel()], axis=1)
    pinv = np.linalg.pinv(design)
    flat = windows.reshape(-1, size * size)
    coefficients = flat @ pinv.T
    residual = flat - coefficients @ design.T
    residual -= residual.mean(axis=1, keepdims=True)
    ra = np.abs(residual).mean(axis=1)
    low = np.partition(residual, 4, axis=1)[:, :5].mean(axis=1)
    high = np.partition(residual, residual.shape[1] - 5, axis=1)[:, -5:].mean(axis=1)
    shape = height_um.shape
    return ra.reshape(shape), (high - low).reshape(shape)


def _local_gloss_map(state, local_ra_um: np.ndarray, size: int) -> np.ndarray:
    """Vectorized established GU terms evaluated over every local window."""
    cfg = GlossProxyConfig()
    init_max = maximum_filter(
        np.asarray(state.initial_scratch_depth_um, dtype=np.float64),
        size=size, mode="reflect")
    residual_max = maximum_filter(
        np.asarray(state.residual_scratch_depth_um, dtype=np.float64),
        size=size, mode="reflect")
    q_scratch = np.ones_like(init_max)
    has_scratch = init_max > cfg.scratch_epsilon_um
    q_scratch[has_scratch] = 1.0 - np.clip(
        residual_max[has_scratch] / init_max[has_scratch], 0.0, 1.0)

    removal = np.asarray(state.cumulative_removal_um, dtype=np.float64)
    removal_mean = uniform_filter(removal, size=size, mode="reflect")
    removal_sq_mean = uniform_filter(removal * removal, size=size, mode="reflect")
    removal_std = np.sqrt(np.maximum(0.0, removal_sq_mean - removal_mean * removal_mean))

    thermal_mean = uniform_filter(
        np.asarray(state.thermal_damage_proxy, dtype=np.float64),
        size=size, mode="reflect")
    q_ra = np.exp(
        -np.maximum(0.0, local_ra_um - cfg.ra_reference_um) / cfg.ra_decay_scale_um)
    q_uniformity = np.exp(-removal_std / cfg.uniformity_decay_scale_um)
    q_thermal = np.exp(-thermal_mean / C.THERMAL_GLOSS_DAMAGE_SCALE)

    weighted = (
        cfg.w_ra * np.log(np.maximum(q_ra, 1e-6))
        + cfg.w_scratch * np.log(np.maximum(q_scratch, 1e-6))
        + cfg.w_uniformity * np.log(np.maximum(q_uniformity, 1e-6))
    )
    weight_sum = cfg.w_ra + cfg.w_scratch + cfg.w_uniformity
    q_total = np.exp(weighted / weight_sum) * np.power(
        np.maximum(q_thermal, 1e-6), cfg.w_thermal)
    return np.asarray(gu_from_relative(q_total, cfg.upper_anchor_gu), dtype=np.float64)


def local_quality_maps(state, targets: AreaQualityTargets | None = None) -> dict[str, np.ndarray]:
    """Build one local quality/pass value per ROI surface cell."""
    targets = targets or AreaQualityTargets()
    _validate(state, targets)
    state.residual_scratch_depth_um = residual_scratch_depth_um(
        state.micro_height_um, state.defect_mask, state.resolution_m)
    size = targets.local_window_cells
    local_ra, local_rz = _local_texture_maps(state.micro_height_um, size)
    local_gu = _local_gloss_map(state, local_ra, size)

    initial_scratch = np.asarray(state.initial_scratch_depth_um, dtype=np.float64)
    residual_scratch = np.asarray(state.residual_scratch_depth_um, dtype=np.float64)
    gu_pass = local_gu >= targets.gu_min
    ra_pass = local_ra <= targets.ra_max_um
    rz_pass = local_rz <= targets.rz_max_um
    scratch_pass = (
        (initial_scratch < targets.scratch_initial_negligible_um)
        | (residual_scratch < initial_scratch)
    )
    all4_pass = gu_pass & ra_pass & rz_pass & scratch_pass

    clearcoat_pass = (
        np.asarray(state.clearcoat_remaining_um, dtype=np.float64)
        >= targets.clearcoat_min_um)
    temperature_pass = (
        np.asarray(state.peak_temperature_c, dtype=np.float64)
        <= targets.temperature_max_c)
    return {
        "local_gu": local_gu,
        "local_ra_um": local_ra,
        "local_rz_um": local_rz,
        "scratch_initial_um": initial_scratch,
        "scratch_residual_um": residual_scratch,
        "gu_pass": gu_pass,
        "ra_pass": ra_pass,
        "rz_pass": rz_pass,
        "scratch_pass": scratch_pass,
        "all4_pass": all4_pass,
        "clearcoat_pass": clearcoat_pass,
        "temperature_pass": temperature_pass,
    }


def _tile_slices(shape: tuple[int, int], tiles: tuple[int, int]):
    xs = np.linspace(0, shape[0], tiles[0] + 1, dtype=int)
    ys = np.linspace(0, shape[1], tiles[1] + 1, dtype=int)
    for i in range(tiles[0]):
        for j in range(tiles[1]):
            yield slice(xs[i], xs[i + 1]), slice(ys[j], ys[j + 1]), i, j


def summarize_area_quality(
    state,
    targets: AreaQualityTargets | None = None,
) -> dict:
    """Return 25 tile rows and ROI summaries without changing pass decisions."""
    targets = targets or AreaQualityTargets()
    maps = local_quality_maps(state, targets)
    pass_names = (
        "gu_pass", "ra_pass", "rz_pass", "scratch_pass", "all4_pass",
        "clearcoat_pass", "temperature_pass",
    )
    tile_rows: list[dict] = []
    for sx, sy, i, j in _tile_slices(
            state.shape, (targets.tiles_x, targets.tiles_y)):
        row = {
            "tile_i": i,
            "tile_j": j,
            "cell_count": int(maps["all4_pass"][sx, sy].size),
            "local_gu_mean": float(maps["local_gu"][sx, sy].mean()),
            "local_ra_mean_um": float(maps["local_ra_um"][sx, sy].mean()),
            "local_rz_mean_um": float(maps["local_rz_um"][sx, sy].mean()),
        }
        for name in pass_names:
            row[f"{name}_area_pct"] = float(maps[name][sx, sy].mean() * 100.0)
        tile_rows.append(row)

    summary = {
        "diagnostic_version": AREA_DIAGNOSTIC_VERSION,
        "design_status": "PT-DESIGN local-window extension; SYNTHETIC output",
        "resolution_m": float(state.resolution_m),
        "local_window_cells": targets.local_window_cells,
        "local_window_m": float(targets.local_window_cells * state.resolution_m),
        "targets": asdict(targets),
    }
    for name in pass_names:
        fractions = np.asarray(
            [row[f"{name}_area_pct"] for row in tile_rows], dtype=np.float64)
        summary[f"roi_{name}_area_pct"] = float(maps[name].mean() * 100.0)
        summary[f"tile_{name}_area_pct_mean"] = float(fractions.mean())
        summary[f"tile_{name}_area_pct_min"] = float(fractions.min())
        summary[f"tile_{name}_area_pct_p10"] = float(np.percentile(fractions, 10))
    for name in ("local_gu", "local_ra_um", "local_rz_um"):
        values = np.asarray(maps[name], dtype=np.float64)
        summary[f"roi_{name}_mean"] = float(values.mean())
        summary[f"roi_{name}_min"] = float(values.min())
        summary[f"roi_{name}_p10"] = float(np.percentile(values, 10))
    if not all(np.isfinite(value) for value in summary.values()
               if isinstance(value, (int, float))):
        raise RuntimeError("area-quality summary contains NaN/Inf")
    return {"summary": summary, "tile_rows": tile_rows, "maps": maps}

