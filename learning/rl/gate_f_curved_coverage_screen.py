"""Gate F3 CPU-only curved geometry and analytic coverage screen.

The established Gate C step-over/path definitions are reused unchanged.  The
screen evaluates nominal geometry, a local-tangent pad footprint, edge support,
normal variation, and 5x5 normalized exposure.  It is a PT-DESIGN analytic
screen, not PhysX contact or predicted physical removal.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os

import numpy as np

from learning.polytwin import config as C
from learning.rl.gate_c_path_coverage_screen import candidate_lines
from learning.rl.gate_f_curved_geometry import (
    SurfaceGeometrySpec,
    build_surface_mesh,
    surface_height_normal,
    validate_surface_mesh,
)


MAP_SIZE_M = (0.32, 0.32)
ROI_SIZE_M = (0.20, 0.20)
ROI_OFFSET_M = tuple(0.5 * (np.asarray(MAP_SIZE_M) - np.asarray(ROI_SIZE_M)))
RESOLUTION_M = 0.002
STEP_OVER_RATIO = 0.40
EDGE_MODE = "balanced_extend5"
DIRECTION_MODES = ("same_xx", "cross_xy")
MESH_MARGIN_M = 0.02
MESH_GRID_SIZE = 41
CURVATURE_RADII_M = (0.30, 0.45, 0.60, 1.00)
FREEFORM_SEEDS = (0, 1, 2)
METRIC_STATUS = "PT-DESIGN_ANALYTIC_GEOMETRY_NOT_PHYSX_OR_PHYSICAL_REMOVAL"


@dataclass(frozen=True)
class GeometryCandidate:
    candidate_id: str
    kind: str
    curvature_radius_m: float
    freeform_seed: int = 0

    @property
    def spec(self) -> SurfaceGeometrySpec:
        return SurfaceGeometrySpec(
            self.kind,
            MAP_SIZE_M,
            self.curvature_radius_m,
            self.freeform_seed,
        )


def geometry_candidates() -> tuple[GeometryCandidate, ...]:
    candidates = [GeometryCandidate("flat", "flat", 1.0e8, 0)]
    for kind in ("cylinder", "sphere"):
        for radius in CURVATURE_RADII_M:
            radius_token = f"{radius:.2f}".replace(".", "p")
            candidates.append(GeometryCandidate(
                f"{kind}_r{radius_token}", kind, radius, 0
            ))
    for seed in FREEFORM_SEEDS:
        candidates.append(GeometryCandidate(f"freeform_s{seed}", "freeform", 0.60, seed))
    return tuple(candidates)


def _surface_points_normals(
    candidate: GeometryCandidate,
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, normal = surface_height_normal(
        candidate.kind,
        candidate.curvature_radius_m,
        MAP_SIZE_M,
        u,
        v,
        freeform_seed=candidate.freeform_seed,
    )
    points = np.stack((u, v, height), axis=-1)
    return np.asarray(height), np.asarray(normal), points


def _sample_path(
    candidate: GeometryCandidate,
    direction_mode: str,
    sample_ds_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    if sample_ds_m <= 0.0:
        raise ValueError("sample_ds_m must be positive")
    _, sweeps = candidate_lines(STEP_OVER_RATIO, EDGE_MODE, direction_mode)
    uv_midpoints: list[np.ndarray] = []
    point_midpoints: list[np.ndarray] = []
    segment_lengths: list[np.ndarray] = []
    path_length_m = 0.0
    line_count = 0
    offset = np.asarray(ROI_OFFSET_M, dtype=np.float64)
    for _, lines in sweeps:
        for start, end in lines:
            start_uv = np.asarray(start, dtype=np.float64) + offset
            end_uv = np.asarray(end, dtype=np.float64) + offset
            length_uv = float(np.linalg.norm(end_uv - start_uv))
            count = max(1, int(np.ceil(length_uv / sample_ds_m)))
            t_edge = np.linspace(0.0, 1.0, count + 1)
            uv_edge = start_uv + (end_uv - start_uv) * t_edge[:, None]
            h_edge, _ = surface_height_normal(
                candidate.kind,
                candidate.curvature_radius_m,
                MAP_SIZE_M,
                uv_edge[:, 0],
                uv_edge[:, 1],
                freeform_seed=candidate.freeform_seed,
            )
            point_edge = np.column_stack((uv_edge, h_edge))
            length_3d = np.linalg.norm(np.diff(point_edge, axis=0), axis=1)
            uv_mid = 0.5 * (uv_edge[:-1] + uv_edge[1:])
            h_mid, _ = surface_height_normal(
                candidate.kind,
                candidate.curvature_radius_m,
                MAP_SIZE_M,
                uv_mid[:, 0],
                uv_mid[:, 1],
                freeform_seed=candidate.freeform_seed,
            )
            uv_midpoints.append(uv_mid)
            point_midpoints.append(np.column_stack((uv_mid, h_mid)))
            segment_lengths.append(length_3d)
            path_length_m += float(length_3d.sum())
            line_count += 1
    return (
        np.concatenate(uv_midpoints, axis=0),
        np.concatenate(point_midpoints, axis=0),
        np.concatenate(segment_lengths, axis=0),
        path_length_m,
        line_count,
    )


def _tile_means(values: np.ndarray) -> np.ndarray:
    if values.shape[0] % 5 or values.shape[1] % 5:
        raise ValueError("coverage grid must divide exactly into 5x5 tiles")
    tile_x = values.shape[0] // 5
    tile_y = values.shape[1] // 5
    out = np.zeros((5, 5), dtype=np.float64)
    for i in range(5):
        for j in range(5):
            out[i, j] = values[i * tile_x:(i + 1) * tile_x,
                               j * tile_y:(j + 1) * tile_y].mean()
    return out


def screen_candidate(
    candidate: GeometryCandidate,
    direction_mode: str,
    *,
    resolution_m: float = RESOLUTION_M,
    coverage_sample_ds_m: float = 0.004,
    geometry_sample_ds_m: float = 0.010,
) -> dict:
    """Evaluate one geometry/path pair without importing Isaac."""
    candidate.spec.validate()
    if direction_mode not in DIRECTION_MODES:
        raise ValueError(f"unknown direction_mode={direction_mode!r}")
    if resolution_m <= 0.0:
        raise ValueError("resolution_m must be positive")
    roi_cells = np.rint(np.asarray(ROI_SIZE_M) / resolution_m).astype(int)
    if np.any(roi_cells % 5):
        raise ValueError("ROI cell counts must divide by 5")
    x = ROI_OFFSET_M[0] + (np.arange(roi_cells[0]) + 0.5) * resolution_m
    y = ROI_OFFSET_M[1] + (np.arange(roi_cells[1]) + 0.5) * resolution_m
    xx, yy = np.meshgrid(x, y, indexing="ij")
    roi_height, roi_normal, roi_points = _surface_points_normals(candidate, xx, yy)

    uv_path, point_path, ds_path, path_length_m, line_count = _sample_path(
        candidate, direction_mode, coverage_sample_ds_m
    )
    _, path_normal = surface_height_normal(
        candidate.kind,
        candidate.curvature_radius_m,
        MAP_SIZE_M,
        uv_path[:, 0],
        uv_path[:, 1],
        freeform_seed=candidate.freeform_seed,
    )
    exposure = np.zeros_like(xx, dtype=np.float64)
    sigma = C.FOOTPRINT_SIGMA_RATIO * C.PAD_RADIUS_M
    projected_radii = C.PAD_RADIUS_M / np.maximum(path_normal[:, 2], 1.0e-6)
    quality_margins = np.minimum.reduce((
        uv_path[:, 0] - projected_radii,
        MAP_SIZE_M[0] - projected_radii - uv_path[:, 0],
        uv_path[:, 1] - projected_radii,
        MAP_SIZE_M[1] - projected_radii - uv_path[:, 1],
    ))
    physical_margins = np.minimum.reduce((
        uv_path[:, 0] + MESH_MARGIN_M - projected_radii,
        MAP_SIZE_M[0] + MESH_MARGIN_M - projected_radii - uv_path[:, 0],
        uv_path[:, 1] + MESH_MARGIN_M - projected_radii,
        MAP_SIZE_M[1] + MESH_MARGIN_M - projected_radii - uv_path[:, 1],
    ))
    for center, normal, ds in zip(point_path, path_normal, ds_path):
        delta = roi_points - center
        normal_distance = np.sum(delta * normal, axis=-1)
        tangent = delta - normal_distance[..., None] * normal
        rho = np.linalg.norm(tangent, axis=-1)
        footprint = np.exp(-0.5 * (rho / sigma) ** 2)
        footprint[rho > C.PAD_RADIUS_M] = 0.0
        exposure += footprint * ds

    mean_exposure = float(exposure.mean())
    normalized = exposure / max(mean_exposure, 1.0e-12)
    tiles = _tile_means(normalized)

    support_resolution = max(resolution_m, 0.004)
    support_x = np.arange(
        -MESH_MARGIN_M + 0.5 * support_resolution,
        MAP_SIZE_M[0] + MESH_MARGIN_M,
        support_resolution,
    )
    support_y = np.arange(
        -MESH_MARGIN_M + 0.5 * support_resolution,
        MAP_SIZE_M[1] + MESH_MARGIN_M,
        support_resolution,
    )
    sx, sy = np.meshgrid(support_x, support_y, indexing="ij")
    support_height, support_normal, support_points = _surface_points_normals(
        candidate, sx, sy
    )
    stride = max(1, int(round(geometry_sample_ds_m / coverage_sample_ds_m)))
    max_tangent_deviation = []
    max_vertical_deviation = []
    max_normal_change_deg = []
    for center, normal in zip(point_path[::stride], path_normal[::stride]):
        delta = support_points - center
        normal_distance = np.sum(delta * normal, axis=-1)
        tangent = delta - normal_distance[..., None] * normal
        tangent_rho = np.linalg.norm(tangent, axis=-1)
        tangent_mask = tangent_rho <= C.PAD_RADIUS_M
        xy_mask = np.hypot(delta[..., 0], delta[..., 1]) <= C.PAD_RADIUS_M
        if not np.any(tangent_mask) or not np.any(xy_mask):
            raise RuntimeError("pad footprint has no support-grid samples")
        max_tangent_deviation.append(float(np.max(np.abs(normal_distance[tangent_mask]))))
        max_vertical_deviation.append(float(np.max(np.abs(delta[..., 2][xy_mask]))))
        cosine = np.clip(np.sum(support_normal * normal, axis=-1), -1.0, 1.0)
        max_normal_change_deg.append(float(np.degrees(np.arccos(cosine[tangent_mask])).max()))

    mesh = build_surface_mesh(
        candidate.spec, grid_size=MESH_GRID_SIZE, margin_m=MESH_MARGIN_M
    )
    mesh_summary = validate_surface_mesh(mesh)
    path_tilt = np.degrees(np.arccos(np.clip(path_normal[:, 2], -1.0, 1.0)))
    support_tilt = np.degrees(np.arccos(np.clip(support_normal[..., 2], -1.0, 1.0)))
    scalars = {
        **asdict(candidate),
        "direction_mode": direction_mode,
        "step_over_ratio": STEP_OVER_RATIO,
        "edge_mode": EDGE_MODE,
        "path_line_count": line_count,
        "path_sample_count": int(len(point_path)),
        "path_length_3d_m": path_length_m,
        "surface_height_min_m": float(support_height.min()),
        "surface_height_max_m": float(support_height.max()),
        "surface_height_range_m": float(support_height.max() - support_height.min()),
        "surface_tilt_max_deg": float(support_tilt.max()),
        "path_tilt_mean_deg": float(path_tilt.mean()),
        "path_tilt_max_deg": float(path_tilt.max()),
        "pad_footprint_normal_change_max_deg": float(max(max_normal_change_deg)),
        "pad_footprint_normal_change_p95_deg": float(np.percentile(max_normal_change_deg, 95)),
        "local_tangent_plane_deviation_max_m": float(max(max_tangent_deviation)),
        "local_tangent_plane_deviation_p95_m": float(np.percentile(max_tangent_deviation, 95)),
        "vertical_pad_height_deviation_max_m": float(max(max_vertical_deviation)),
        "minimum_quality_map_projected_margin_m": float(quality_margins.min()),
        "minimum_physical_mesh_projected_margin_m": float(physical_margins.min()),
        "quality_map_projected_footprint_inside": bool(quality_margins.min() >= -1.0e-12),
        "physical_mesh_projected_footprint_inside": bool(physical_margins.min() >= -1.0e-12),
        "coverage_any_fraction": float((exposure > 0.0).mean()),
        "coverage_effective_fraction": float((normalized >= 0.20).mean()),
        "exposure_cv": float(normalized.std()),
        "exposure_min_norm": float(normalized.min()),
        "exposure_p05_norm": float(np.percentile(normalized, 5)),
        "exposure_p95_norm": float(np.percentile(normalized, 95)),
        "tile_exposure_min_norm": float(tiles.min()),
        "tile_exposure_max_norm": float(tiles.max()),
        "tile_exposure_range_norm": float(tiles.max() - tiles.min()),
        "mesh_vertex_count": int(mesh_summary["vertex_count"]),
        "mesh_triangle_count": int(mesh_summary["triangle_count"]),
        "mesh_minimum_face_normal_z": float(mesh_summary["minimum_face_normal_z"]),
        "all_finite": True,
        "metric_status": METRIC_STATUS,
    }
    numeric_values = [
        value for value in scalars.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    scalars["all_finite"] = bool(np.isfinite(numeric_values).all())
    return {
        "scalars": scalars,
        "normalized_exposure": normalized,
        "tile_exposure_norm": tiles,
    }


def select_f4_candidates(rows: list[dict]) -> list[dict]:
    """Select a representative median geometry per family, then best direction.

    This is a PT-DESIGN diagnostic selection, not a claim that the chosen
    geometry is physically optimal.  Extreme curvature is retained in the CSV
    but not sent to the first PhysX smoke.
    """
    selected: list[dict] = []
    for kind in ("flat", "cylinder", "sphere", "freeform"):
        kind_rows = [
            row for row in rows
            if row["kind"] == kind
            and row["all_finite"]
            and row["physical_mesh_projected_footprint_inside"]
            and row["coverage_effective_fraction"] >= 0.99
        ]
        if not kind_rows:
            continue
        best_per_geometry: list[dict] = []
        for candidate_id in sorted({row["candidate_id"] for row in kind_rows}):
            variants = [row for row in kind_rows if row["candidate_id"] == candidate_id]
            variants.sort(key=lambda row: (
                row["exposure_cv"],
                row["tile_exposure_range_norm"],
                row["path_length_3d_m"],
                row["direction_mode"],
            ))
            best_per_geometry.append(variants[0])
        best_per_geometry.sort(key=lambda row: (
            row["surface_tilt_max_deg"], row["candidate_id"]
        ))
        representative = best_per_geometry[(len(best_per_geometry) - 1) // 2]
        selected.append({
            **representative,
            "selection_status": "PT-DESIGN_REPRESENTATIVE_MEDIAN_GEOMETRY_FOR_F4",
        })
    return selected


def run_screen() -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    tile_rows: list[dict] = []
    for candidate in geometry_candidates():
        for direction_mode in DIRECTION_MODES:
            result = screen_candidate(candidate, direction_mode)
            row = result["scalars"]
            rows.append(row)
            for i, j in np.ndindex((5, 5)):
                tile_rows.append({
                    "candidate_id": candidate.candidate_id,
                    "kind": candidate.kind,
                    "curvature_radius_m": candidate.curvature_radius_m,
                    "freeform_seed": candidate.freeform_seed,
                    "direction_mode": direction_mode,
                    "tile_x": i,
                    "tile_y": j,
                    "exposure_norm": float(result["tile_exposure_norm"][i, j]),
                    "metric_status": METRIC_STATUS,
                })
    rows.sort(key=lambda row: (
        not row["physical_mesh_projected_footprint_inside"],
        not row["all_finite"],
        -row["coverage_effective_fraction"],
        row["pad_footprint_normal_change_max_deg"],
        row["local_tangent_plane_deviation_max_m"],
        row["exposure_cv"],
    ))
    for rank, row in enumerate(rows, 1):
        row["geometry_rank"] = rank
    selected = select_f4_candidates(rows)
    return rows, tile_rows, selected


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite existing output directory: {out_dir}")
    os.makedirs(out_dir)

    rows, tile_rows, selected = run_screen()
    _write_csv(os.path.join(out_dir, "geometry_candidates.csv"), rows)
    _write_csv(os.path.join(out_dir, "coverage_tiles.csv"), tile_rows)
    _write_csv(os.path.join(out_dir, "selected_candidates.csv"), selected)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F3",
        "candidate_geometry_count": len(geometry_candidates()),
        "candidate_path_count": len(rows),
        "tile_row_count": len(tile_rows),
        "selected_for_f4_count": len(selected),
        "map_size_m": MAP_SIZE_M,
        "roi_size_m": ROI_SIZE_M,
        "resolution_m": RESOLUTION_M,
        "pad_radius_m": C.PAD_RADIUS_M,
        "step_over_ratio": STEP_OVER_RATIO,
        "edge_mode": EDGE_MODE,
        "direction_modes": DIRECTION_MODES,
        "curvature_radii_m": CURVATURE_RADII_M,
        "freeform_seeds": FREEFORM_SEEDS,
        "mesh_margin_m": MESH_MARGIN_M,
        "mesh_grid_size": MESH_GRID_SIZE,
        "metric_status": METRIC_STATUS,
        "physx_executed": False,
        "training_executed": False,
        "selection_rule": (
            "per family: finite + physical mesh support + effective coverage >=0.99; "
            "choose median surface tilt geometry and its lowest-CV direction"
        ),
        "selected_candidates": selected,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"[Gate F3] wrote {out_dir}")
    for row in selected:
        print(
            f"selected {row['candidate_id']} {row['direction_mode']} "
            f"tilt={row['surface_tilt_max_deg']:.3f}deg "
            f"normal_change={row['pad_footprint_normal_change_max_deg']:.3f}deg "
            f"coverage={row['coverage_effective_fraction']:.5f} "
            f"physical_margin={row['minimum_physical_mesh_projected_margin_m']:.5f}m"
        )


if __name__ == "__main__":
    main()

