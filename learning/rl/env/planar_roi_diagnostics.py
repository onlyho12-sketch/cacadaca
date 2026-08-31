"""Large continuous planar map + centered evaluation-ROI diagnostics.

This module is deliberately Isaac-free so geometry and roughness/removal
decomposition can be unit-tested with the system Python.  Every cutoff and
under/over threshold here is PT-DESIGN; the outputs are synthetic diagnostics,
not profilometer measurements.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from learning.polytwin import config as PC
from learning.polytwin.path_executor import raster_waypoints
from learning.polytwin.roughness_metrics import (
    ra_um,
    residual_scratch_depth_um,
    rz_um,
)
from learning.polytwin.surface_state import SurfaceState, make_flat_patch


@dataclass(frozen=True)
class PlanarRoiGeometry:
    map_size_m: tuple[float, float] = (0.32, 0.32)
    roi_size_m: tuple[float, float] = (0.20, 0.20)
    resolution_m: float = 0.002
    pad_radius_m: float = PC.PAD_RADIUS_M
    workpiece_extra_m: float = 0.04

    def validate(self) -> dict[str, float | bool]:
        map_margin_x = 0.5 * (self.map_size_m[0] - self.roi_size_m[0])
        map_margin_y = 0.5 * (self.map_size_m[1] - self.roi_size_m[1])
        workpiece_size_x = self.map_size_m[0] + self.workpiece_extra_m
        workpiece_size_y = self.map_size_m[1] + self.workpiece_extra_m
        workpiece_margin_x = 0.5 * (workpiece_size_x - self.roi_size_m[0])
        workpiece_margin_y = 0.5 * (workpiece_size_y - self.roi_size_m[1])
        out = {
            "map_margin_x_m": map_margin_x,
            "map_margin_y_m": map_margin_y,
            "quality_footprint_margin_x_m": map_margin_x - self.pad_radius_m,
            "quality_footprint_margin_y_m": map_margin_y - self.pad_radius_m,
            "workpiece_size_x_m": workpiece_size_x,
            "workpiece_size_y_m": workpiece_size_y,
            "physical_footprint_margin_x_m": workpiece_margin_x - self.pad_radius_m,
            "physical_footprint_margin_y_m": workpiece_margin_y - self.pad_radius_m,
        }
        out["quality_footprint_inside"] = bool(
            out["quality_footprint_margin_x_m"] >= -1e-12
            and out["quality_footprint_margin_y_m"] >= -1e-12
        )
        out["physical_footprint_inside"] = bool(
            out["physical_footprint_margin_x_m"] >= -1e-12
            and out["physical_footprint_margin_y_m"] >= -1e-12
        )
        for full, roi, name in zip(self.map_size_m, self.roi_size_m, ("x", "y")):
            if full <= roi:
                raise ValueError(f"map {name} must be larger than ROI: {full} <= {roi}")
            full_cells = full / self.resolution_m
            roi_cells = roi / self.resolution_m
            if abs(full_cells - round(full_cells)) > 1e-9:
                raise ValueError(f"map {name} is not an integer number of cells")
            if abs(roi_cells - round(roi_cells)) > 1e-9:
                raise ValueError(f"ROI {name} is not an integer number of cells")
            if (round(full_cells) - round(roi_cells)) % 2:
                raise ValueError(f"ROI {name} cannot be centered on cell boundaries")
        if not out["quality_footprint_inside"]:
            raise ValueError("pad footprint leaves the quality map at an ROI path endpoint")
        if not out["physical_footprint_inside"]:
            raise ValueError("pad footprint leaves the physical workpiece at an ROI path endpoint")
        return out


def centered_roi_slices(geometry: PlanarRoiGeometry) -> tuple[slice, slice]:
    geometry.validate()
    full = np.rint(np.asarray(geometry.map_size_m) / geometry.resolution_m).astype(int)
    roi = np.rint(np.asarray(geometry.roi_size_m) / geometry.resolution_m).astype(int)
    start = (full - roi) // 2
    return (slice(int(start[0]), int(start[0] + roi[0])),
            slice(int(start[1]), int(start[1] + roi[1])))


def centered_raster_lines(geometry: PlanarRoiGeometry, spacing_m: float,
                          direction: str = "x") -> list[tuple[tuple[float, float],
                                                                  tuple[float, float], int]]:
    """Raster lines in full-map UV coordinates, centered on the evaluation ROI."""
    geometry.validate()
    off_x = 0.5 * (geometry.map_size_m[0] - geometry.roi_size_m[0])
    off_y = 0.5 * (geometry.map_size_m[1] - geometry.roi_size_m[1])
    local = raster_waypoints(geometry.roi_size_m, spacing_m)
    lines = []
    for p0, p1, line_id in local:
        if direction == "x":
            q0 = (p0[0] + off_x, p0[1] + off_y)
            q1 = (p1[0] + off_x, p1[1] + off_y)
        elif direction == "y":
            q0 = (p0[1] + off_x, p0[0] + off_y)
            q1 = (p1[1] + off_x, p1[0] + off_y)
        else:
            raise ValueError(f"unknown raster direction: {direction}")
        lines.append((q0, q1, line_id))
    return lines


def surface_view(state: SurfaceState, slices: tuple[slice, slice]) -> SurfaceState:
    """Shallow SurfaceState copy whose spatial arrays are ROI views of the original."""
    view = copy.copy(state)
    shape = state.shape
    for name, value in vars(state).items():
        if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[:2] == shape:
            setattr(view, name, value[slices])
    return view


def make_centered_roi_patch(geometry: PlanarRoiGeometry, seed: int) -> SurfaceState:
    """Create a continuous map while keeping ROI defect density independent of surround size.

    The surround is a scratch-free support field.  The central ROI is generated with the
    requested seed and normal scratch count, then inserted into the support map.  This avoids
    diluting 4--12 scratches over 320 mm when only the central 200 mm is evaluated.
    """
    sl = centered_roi_slices(geometry)
    support_seed = seed + 1_000_003
    outer = make_flat_patch(
        geometry.map_size_m, geometry.resolution_m, seed=support_seed, with_scratches=False,
    )
    defect_source = make_flat_patch(
        geometry.roi_size_m, geometry.resolution_m, seed=seed, with_scratches=True,
    )
    # Keep the support map's roughness and clearcoat fields untouched across the
    # ROI boundary.  Only the ROI-local scratch grooves/masks are inserted, so
    # there is no artificial clearcoat or background-height seam.
    scratches = defect_source.initial_scratch_depth_um
    outer.micro_height_um[sl] -= scratches
    outer.initial_micro_height_um[sl] -= scratches
    outer.initial_scratch_depth_um[sl] = scratches
    outer.residual_scratch_depth_um[sl] = scratches
    outer.defect_mask[sl] = defect_source.defect_mask
    outer.healthy_mask[sl] = ~defect_source.defect_mask
    return outer


def _tile_slices(shape: tuple[int, int], tiles: tuple[int, int]):
    xs = np.linspace(0, shape[0], tiles[0] + 1, dtype=int)
    ys = np.linspace(0, shape[1], tiles[1] + 1, dtype=int)
    for i in range(tiles[0]):
        for j in range(tiles[1]):
            yield slice(xs[i], xs[i + 1]), slice(ys[j], ys[j + 1]), i, j


def _texture_metrics(height_um: np.ndarray) -> tuple[float, float]:
    return ra_um(height_um), rz_um(height_um)


def diagnose_roi(state: SurfaceState, geometry: PlanarRoiGeometry,
                 waviness_sigma_m: float = 0.010,
                 under_over_fraction: float = 0.20,
                 center_fraction: float = 0.50,
                 edge_band_fraction: float = 0.20,
                 tiles: tuple[int, int] = (5, 5)) -> dict:
    """Separate total texture, fine texture, and low-frequency removal waviness.

    `waviness_sigma_m` is a PT-DESIGN Gaussian scale.  It is not an ISO roughness
    cutoff.  The split is used only to test whether raster-scale removal topography
    is contaminating the existing synthetic Ra/Rz score.
    """
    if waviness_sigma_m <= 0.0:
        raise ValueError("waviness_sigma_m must be positive")
    if not (0.0 < under_over_fraction < 1.0):
        raise ValueError("under_over_fraction must be in (0, 1)")
    sl = centered_roi_slices(geometry)
    roi = surface_view(state, sl)
    sigma_cells = waviness_sigma_m / geometry.resolution_m

    height = np.asarray(roi.micro_height_um, dtype=float)
    removal = np.asarray(roi.cumulative_removal_um, dtype=float)
    low_height = gaussian_filter(height, sigma=sigma_cells, mode="reflect")
    fine_height = height - low_height
    removal_low = gaussian_filter(removal, sigma=sigma_cells, mode="reflect")
    removal_fine = removal - removal_low

    roi.residual_scratch_depth_um[:] = residual_scratch_depth_um(
        height, roi.defect_mask, roi.resolution_m)
    total_ra, total_rz = _texture_metrics(height)
    fine_ra, fine_rz = _texture_metrics(fine_height)
    waviness_ra, waviness_rz = _texture_metrics(-removal_low)

    mean_removal = float(removal.mean())
    std_removal = float(removal.std())
    if mean_removal > 1e-12:
        under = removal < mean_removal * (1.0 - under_over_fraction)
        over = removal > mean_removal * (1.0 + under_over_fraction)
        cv = std_removal / mean_removal
    else:
        under = np.zeros_like(removal, dtype=bool)
        over = np.zeros_like(removal, dtype=bool)
        cv = 0.0

    nx, ny = removal.shape
    cx = max(1, int(round(nx * center_fraction)))
    cy = max(1, int(round(ny * center_fraction)))
    x0, y0 = (nx - cx) // 2, (ny - cy) // 2
    center_mask = np.zeros_like(removal, dtype=bool)
    center_mask[x0:x0 + cx, y0:y0 + cy] = True
    bx = max(1, int(round(nx * edge_band_fraction)))
    by = max(1, int(round(ny * edge_band_fraction)))
    edge_mask = np.zeros_like(removal, dtype=bool)
    edge_mask[:bx, :] = True; edge_mask[-bx:, :] = True
    edge_mask[:, :by] = True; edge_mask[:, -by:] = True
    center_mean = float(removal[center_mask].mean())
    edge_mean = float(removal[edge_mask].mean())

    scalars = {
        "roi_total_ra_um": total_ra,
        "roi_total_rz_um": total_rz,
        "roi_fine_ra_um": fine_ra,
        "roi_fine_rz_um": fine_rz,
        "roi_ra_low_frequency_contribution_um": total_ra - fine_ra,
        "roi_rz_low_frequency_contribution_um": total_rz - fine_rz,
        "roi_removal_mean_um": mean_removal,
        "roi_removal_std_um": std_removal,
        "roi_removal_cv": cv,
        "roi_removal_max_min_um": float(removal.max() - removal.min()),
        "roi_removal_p95_p05_um": float(np.percentile(removal, 95)
                                          - np.percentile(removal, 5)),
        "roi_removal_waviness_ra_um": waviness_ra,
        "roi_removal_waviness_rz_um": waviness_rz,
        "roi_removal_waviness_std_um": float(removal_low.std()),
        "roi_removal_fine_std_um": float(removal_fine.std()),
        "roi_center_removal_mean_um": center_mean,
        "roi_edge_removal_mean_um": edge_mean,
        "roi_center_edge_delta_um": center_mean - edge_mean,
        "roi_center_edge_ratio": center_mean / max(edge_mean, 1e-12),
        "roi_under_fraction": float(under.mean()),
        "roi_over_fraction": float(over.mean()),
        "roi_coverage_fraction": float(
            (removal >= PC.MINIMUM_EFFECTIVE_REMOVAL_UM).mean()),
        "roi_scratch_mean_um": float(roi.residual_scratch_depth_um.mean()),
        "roi_scratch_max_um": float(roi.residual_scratch_depth_um.max()),
        "roi_clearcoat_min_um": float(roi.clearcoat_remaining_um.min()),
        "waviness_sigma_m": float(waviness_sigma_m),
        "under_over_fraction": float(under_over_fraction),
    }

    maps = {name: np.zeros(tiles, dtype=float) for name in (
        "removal_mean_um", "removal_std_um", "total_ra_um", "total_rz_um",
        "fine_ra_um", "fine_rz_um", "waviness_std_um", "scratch_mean_um",
        "scratch_max_um", "clearcoat_min_um", "under_fraction", "over_fraction",
        "coverage_fraction",
    )}
    for sx, sy, i, j in _tile_slices(removal.shape, tiles):
        h = height[sx, sy]
        fh = fine_height[sx, sy]
        rem = removal[sx, sy]
        scr = roi.residual_scratch_depth_um[sx, sy]
        maps["removal_mean_um"][i, j] = rem.mean()
        maps["removal_std_um"][i, j] = rem.std()
        maps["total_ra_um"][i, j], maps["total_rz_um"][i, j] = _texture_metrics(h)
        maps["fine_ra_um"][i, j], maps["fine_rz_um"][i, j] = _texture_metrics(fh)
        maps["waviness_std_um"][i, j] = removal_low[sx, sy].std()
        maps["scratch_mean_um"][i, j] = scr.mean()
        maps["scratch_max_um"][i, j] = scr.max()
        maps["clearcoat_min_um"][i, j] = roi.clearcoat_remaining_um[sx, sy].min()
        maps["under_fraction"][i, j] = under[sx, sy].mean()
        maps["over_fraction"][i, j] = over[sx, sy].mean()
        maps["coverage_fraction"][i, j] = (
            rem >= PC.MINIMUM_EFFECTIVE_REMOVAL_UM).mean()

    return {"scalars": scalars, "tile_maps": maps,
            "geometry": geometry.validate(), "roi_slices": sl}
