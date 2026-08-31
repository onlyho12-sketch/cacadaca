"""Gate C1 analytic geometry/coverage screen (no Isaac, no training).

The screen integrates the established Gaussian pad footprint along candidate
raster centerlines.  It compares normalized exposure geometry only; outputs are
PT-DESIGN diagnostics, not predicted physical removal or measured coverage.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone

import numpy as np

from learning.polytwin import config as C
from learning.polytwin.path_executor import raster_waypoints


ROI_SIZE_M = (0.20, 0.20)
MAP_SIZE_M = (0.32, 0.32)
RESOLUTION_M = 0.002
ENDPOINT_EXTENSION_M = 0.005
STEP_OVER_RATIOS = (0.18, 0.25, 0.32, 0.40)
EDGE_MODES = ("legacy_start", "balanced", "balanced_extend5")
DIRECTIONS = ("same_xx", "cross_xy")


def _local_lines(spacing_m: float, edge_mode: str, direction: str):
    legacy = raster_waypoints(ROI_SIZE_M, spacing_m)
    n_lines = len(legacy)
    if edge_mode == "legacy_start":
        offsets = [float(p0[1]) for p0, _, _ in legacy]
        extension = 0.0
    elif edge_mode in {"balanced", "balanced_extend5"}:
        span = (n_lines - 1) * spacing_m
        margin = 0.5 * (ROI_SIZE_M[1] - span)
        offsets = [margin + i * spacing_m for i in range(n_lines)]
        extension = ENDPOINT_EXTENSION_M if edge_mode == "balanced_extend5" else 0.0
    else:
        raise ValueError(edge_mode)
    lines = []
    for line_id, offset in enumerate(offsets):
        p0 = (-extension, offset)
        p1 = (ROI_SIZE_M[0] + extension, offset)
        if line_id % 2:
            p0, p1 = p1, p0
        if direction == "y":
            p0 = (p0[1], p0[0]); p1 = (p1[1], p1[0])
        lines.append((p0, p1))
    return lines


def candidate_lines(step_over_ratio: float, edge_mode: str, direction_mode: str):
    spacing = step_over_ratio * C.PAD_DIAMETER_M
    x_lines = _local_lines(spacing, edge_mode, "x")
    if direction_mode == "same_xx":
        sweeps = [("x", x_lines), ("x", x_lines)]
    elif direction_mode == "cross_xy":
        sweeps = [("x", x_lines), ("y", _local_lines(spacing, edge_mode, "y"))]
    else:
        raise ValueError(direction_mode)
    return spacing, sweeps


def integrate_exposure(step_over_ratio: float, edge_mode: str,
                       direction_mode: str, sample_ds_m: float = 0.002) -> dict:
    spacing, sweeps = candidate_lines(step_over_ratio, edge_mode, direction_mode)
    nx = int(round(ROI_SIZE_M[0] / RESOLUTION_M))
    ny = int(round(ROI_SIZE_M[1] / RESOLUTION_M))
    x = (np.arange(nx) + 0.5) * RESOLUTION_M
    y = (np.arange(ny) + 0.5) * RESOLUTION_M
    xx, yy = np.meshgrid(x, y, indexing="ij")
    exposure = np.zeros((nx, ny), dtype=float)
    path_length = 0.0
    path_centers = []
    sigma = C.FOOTPRINT_SIGMA_RATIO * C.PAD_RADIUS_M
    for _, lines in sweeps:
        for p0, p1 in lines:
            p0 = np.asarray(p0, dtype=float); p1 = np.asarray(p1, dtype=float)
            length = float(np.linalg.norm(p1 - p0))
            n = max(1, int(np.ceil(length / sample_ds_m)))
            ds = length / n
            direction = (p1 - p0) / max(length, 1e-12)
            for k in range(n):
                center = p0 + direction * ((k + 0.5) * ds)
                rho = np.hypot(xx - center[0], yy - center[1])
                footprint = np.exp(-0.5 * (rho / sigma) ** 2)
                footprint[rho > C.PAD_RADIUS_M] = 0.0
                exposure += footprint * ds
                path_centers.append(center)
            path_length += length
    centers = np.asarray(path_centers)
    map_offset = 0.5 * (np.asarray(MAP_SIZE_M) - np.asarray(ROI_SIZE_M))
    centers_full = centers + map_offset
    quality_margin = np.minimum(
        centers_full - C.PAD_RADIUS_M,
        np.asarray(MAP_SIZE_M) - C.PAD_RADIUS_M - centers_full)
    footprint_inside = bool(quality_margin.min() >= -1e-12)

    mean = float(exposure.mean())
    normalized = exposure / max(mean, 1e-12)
    center_mask = np.zeros_like(exposure, dtype=bool)
    center_mask[25:75, 25:75] = True
    edge_mask = np.zeros_like(exposure, dtype=bool)
    edge_mask[:20, :] = True; edge_mask[-20:, :] = True
    edge_mask[:, :20] = True; edge_mask[:, -20:] = True
    corner_mask = np.zeros_like(exposure, dtype=bool)
    corner_mask[:20, :20] = True; corner_mask[:20, -20:] = True
    corner_mask[-20:, :20] = True; corner_mask[-20:, -20:] = True
    center_mean = float(normalized[center_mask].mean())
    edge_mean = float(normalized[edge_mask].mean())
    coverage_any = float((exposure > 0.0).mean())
    coverage_effective = float((normalized >= 0.20).mean())
    scalars = {
        "step_over_ratio": step_over_ratio,
        "step_over_m": spacing,
        "edge_mode": edge_mode,
        "direction_mode": direction_mode,
        "lines_per_sweep": len(sweeps[0][1]),
        "sweeps": len(sweeps),
        "path_length_m": path_length,
        "footprint_inside_quality_map": footprint_inside,
        "minimum_quality_map_footprint_margin_m": float(quality_margin.min()),
        "coverage_any_fraction": coverage_any,
        "coverage_effective_fraction": coverage_effective,
        "exposure_cv": float(normalized.std()),
        "exposure_min_norm": float(normalized.min()),
        "exposure_p05_norm": float(np.percentile(normalized, 5)),
        "exposure_p95_norm": float(np.percentile(normalized, 95)),
        "exposure_p95_p05_norm": float(np.percentile(normalized, 95)
                                          - np.percentile(normalized, 5)),
        "under_80pct_fraction": float((normalized < 0.80).mean()),
        "over_120pct_fraction": float((normalized > 1.20).mean()),
        "center_mean_norm": center_mean,
        "edge_mean_norm": edge_mean,
        "center_edge_delta_norm": center_mean - edge_mean,
        "center_edge_ratio": center_mean / max(edge_mean, 1e-12),
        "edge_min_norm": float(normalized[edge_mask].min()),
        "corner_mean_norm": float(normalized[corner_mask].mean()),
        "corner_min_norm": float(normalized[corner_mask].min()),
        "metric_status": "PT-DESIGN_ANALYTIC_EXPOSURE_NOT_PHYSICAL_REMOVAL",
    }
    tiles = np.zeros((5, 5), dtype=float)
    for i in range(5):
        for j in range(5):
            tiles[i, j] = normalized[i * 20:(i + 1) * 20,
                                     j * 20:(j + 1) * 20].mean()
    return {"scalars": scalars, "normalized_exposure": normalized,
            "tile_exposure_norm": tiles}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite existing output directory: {out_dir}")
    os.makedirs(out_dir)

    rows, tile_rows = [], []
    for ratio in STEP_OVER_RATIOS:
        for edge_mode in EDGE_MODES:
            for direction in DIRECTIONS:
                result = integrate_exposure(ratio, edge_mode, direction)
                row = result["scalars"]
                rows.append(row)
                for i, j in np.ndindex((5, 5)):
                    tile_rows.append({
                        "step_over_ratio": ratio, "edge_mode": edge_mode,
                        "direction_mode": direction, "tile_x": i, "tile_y": j,
                        "exposure_norm": float(result["tile_exposure_norm"][i, j]),
                    })
    rows.sort(key=lambda row: (
        not row["footprint_inside_quality_map"],
        -row["coverage_effective_fraction"], row["exposure_cv"],
        abs(row["center_edge_delta_norm"]), row["path_length_m"]))
    for rank, row in enumerate(rows, 1):
        row["uniformity_rank"] = rank

    with open(os.path.join(out_dir, "analytic_candidates.csv"),
              "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(out_dir, "analytic_tiles.csv"),
              "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tile_rows[0]))
        writer.writeheader(); writer.writerows(tile_rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(rows), "step_over_ratios": STEP_OVER_RATIOS,
        "edge_modes": EDGE_MODES, "direction_modes": DIRECTIONS,
        "roi_size_m": ROI_SIZE_M, "map_size_m": MAP_SIZE_M,
        "resolution_m": RESOLUTION_M, "pad_radius_m": C.PAD_RADIUS_M,
        "footprint_sigma_ratio": C.FOOTPRINT_SIGMA_RATIO,
        "endpoint_extension_m": ENDPOINT_EXTENSION_M,
        "effective_coverage_threshold_relative_to_mean": 0.20,
        "metric_status": "PT-DESIGN_ANALYTIC_EXPOSURE_NOT_PHYSICAL_REMOVAL",
        "ranked_candidates": rows,
    }
    with open(os.path.join(out_dir, "analytic_summary.json"),
              "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"[Gate C1] wrote {out_dir}")
    for row in rows[:8]:
        print(
            f"rank={row['uniformity_rank']:2d} so={row['step_over_ratio']:.2f} "
            f"{row['edge_mode']:16s} {row['direction_mode']:8s} "
            f"cv={row['exposure_cv']:.4f} c/e={row['center_edge_ratio']:.4f} "
            f"under={row['under_80pct_fraction']:.3f} path={row['path_length_m']:.2f}m")


if __name__ == "__main__":
    main()
