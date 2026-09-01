"""Gate F10-B vehicle point-cloud geometry adapter (CPU only).

The adapter reads the canonical vehicle point cloud and existing C/SL/SR path
arrays without modifying them.  A robust local quadratic fit provides outward
normals, signed principal curvatures, pad-scale risk and fit/boundary evidence.
Thresholds in this isolated gate are PT-DESIGN diagnostics, not production
safety limits.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

RAILS = ("C", "SL", "SR")


@dataclass(frozen=True)
class GeometryConfig:
    fit_radius_m: float = 0.060
    support_radius_m: float = 0.075
    pad_radius_m: float = 0.055
    min_neighbors: int = 24
    max_neighbors: int = 320
    robust_iterations: int = 4
    huber_delta: float = 1.5
    flat_curvature_1_m: float = 0.05
    confidence_review_threshold: float = 0.55
    confidence_unsafe_threshold: float = 0.40
    boundary_unsafe_risk: float = 0.85
    normal_spread_unsafe_deg: float = 20.0
    curvature_unsafe_ratio: float = 0.80
    path_deviation_unsafe_m: float = 0.020
    path_continuity_max_step_m: float = 0.080


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ascii_xyz_ply(path: str) -> np.ndarray:
    """Load an ASCII PLY containing x/y/z as its first three vertex fields."""
    vertex_count = None
    header_lines = 0
    is_ascii = False
    properties = []
    with open(path, "r", encoding="ascii") as fh:
        in_vertex = False
        for line in fh:
            header_lines += 1
            words = line.strip().split()
            if words[:2] == ["format", "ascii"]:
                is_ascii = True
            elif words[:2] == ["element", "vertex"]:
                vertex_count = int(words[2])
                in_vertex = True
            elif words and words[0] == "element" and words[1] != "vertex":
                in_vertex = False
            elif in_vertex and words[:1] == ["property"]:
                properties.append(words[-1])
            elif words[:1] == ["end_header"]:
                break
    if not is_ascii or vertex_count is None:
        raise ValueError(f"only ASCII vertex PLY is supported: {path}")
    if properties[:3] != ["x", "y", "z"]:
        raise ValueError(f"first PLY vertex properties must be x/y/z: {properties[:3]}")
    points = np.loadtxt(path, skiprows=header_lines, max_rows=vertex_count,
                        usecols=(0, 1, 2), dtype=np.float64)
    if points.shape != (vertex_count, 3) or not np.isfinite(points).all():
        raise ValueError(f"invalid PLY vertex data: expected {(vertex_count, 3)}, got {points.shape}")
    return points


def segment_files(scan_dir: str, rail: str) -> list[tuple[int, str]]:
    rows = []
    for path in glob.glob(os.path.join(scan_dir, f"path_{rail}*.npy")):
        match = re.search(rf"path_{rail}(\d+)\.npy$", os.path.basename(path))
        if match:
            rows.append((int(match.group(1)), path))
    return sorted(rows)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("zero-length vector")
    return vector / norm


def _robust_quadratic(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                      scale_m: float, cfg: GeometryConfig) -> tuple[np.ndarray, float, float]:
    xs, ys = x / scale_m, y / scale_m
    design = np.column_stack((np.ones(len(x)), xs, ys, 0.5 * xs * xs,
                              xs * ys, 0.5 * ys * ys))
    weights = np.ones(len(x), dtype=np.float64)
    coeff = np.zeros(6, dtype=np.float64)
    for _ in range(cfg.robust_iterations):
        weighted = np.sqrt(weights)
        coeff = np.linalg.lstsq(design * weighted[:, None], z * weighted, rcond=None)[0]
        residual = z - design @ coeff
        median = float(np.median(residual))
        sigma = 1.4826 * float(np.median(np.abs(residual - median))) + 1e-9
        scaled = np.abs(residual - median) / (cfg.huber_delta * sigma)
        weights = np.ones_like(scaled)
        outliers = scaled > 1.0
        weights[outliers] = 1.0 / scaled[outliers]
    residual = z - design @ coeff
    rms = float(np.sqrt(np.average(residual * residual, weights=weights)))
    condition = float(np.linalg.cond(design * np.sqrt(weights)[:, None]))
    # Convert derivatives from normalized coordinates back to metres.
    coeff_metric = coeff.copy()
    coeff_metric[1:3] /= scale_m
    coeff_metric[3:] /= scale_m * scale_m
    return coeff_metric, rms, condition


def _principal_curvatures(coeff: np.ndarray) -> tuple[float, float]:
    fx, fy = float(coeff[1]), float(coeff[2])
    fxx, fxy, fyy = float(coeff[3]), float(coeff[4]), float(coeff[5])
    first = np.array([[1.0 + fx * fx, fx * fy],
                      [fx * fy, 1.0 + fy * fy]], dtype=np.float64)
    second = np.array([[fxx, fxy], [fxy, fyy]], dtype=np.float64) / np.sqrt(
        1.0 + fx * fx + fy * fy)
    shape = np.linalg.solve(first, second)
    values = np.linalg.eigvals(shape).real
    values.sort()
    return float(values[1]), float(values[0])


def _classification(k1: float, k2: float, flat: float) -> str:
    if max(abs(k1), abs(k2)) < flat:
        return "near_flat"
    if k1 * k2 < -(flat * flat):
        return "saddle"
    if k1 <= flat and k2 < -flat:
        return "convex"
    if k2 >= -flat and k1 > flat:
        return "concave"
    return "cylindrical_or_transition"


def _boundary_evidence(xy: np.ndarray, pad_radius_m: float) -> tuple[float, float, float]:
    radius = np.linalg.norm(xy, axis=1)
    useful = radius > 1e-8
    if useful.sum() < 6:
        return 0.0, 0.0, 1.0
    angles = np.mod(np.arctan2(xy[useful, 1], xy[useful, 0]), 2.0 * np.pi)
    bins = np.unique(np.floor(angles / (2.0 * np.pi / 16.0)).astype(int))
    angular_coverage = len(bins) / 16.0
    ordered = np.sort(angles)
    gaps = np.diff(np.r_[ordered, ordered[0] + 2.0 * np.pi])
    gap_risk = float(np.clip((float(gaps.max()) - np.pi / 4.0) / (3.0 * np.pi / 4.0), 0.0, 1.0))
    boundary_distance = 0.0
    try:
        hull = ConvexHull(xy)
        # Hull equation a*x+b*y+c <= 0; query is local origin.
        distances = -hull.equations[:, 2] / np.linalg.norm(hull.equations[:, :2], axis=1)
        boundary_distance = max(0.0, float(distances.min()))
    except QhullError:
        pass
    boundary_risk = float(np.clip(1.0 - boundary_distance / pad_radius_m, 0.0, 1.0))
    hole_risk = max(1.0 - angular_coverage, gap_risk)
    return boundary_distance, angular_coverage, float(hole_risk)


def fit_local_surface(query: np.ndarray, neighbors: np.ndarray, cloud_center: np.ndarray,
                      cfg: GeometryConfig) -> dict:
    if len(neighbors) < cfg.min_neighbors:
        raise ValueError(f"need at least {cfg.min_neighbors} neighbors, got {len(neighbors)}")
    centered = neighbors - neighbors.mean(axis=0)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal0 = _unit(eigenvectors[:, 0])
    outward_hint = query - cloud_center
    if np.dot(normal0, outward_hint) < 0.0:
        normal0 = -normal0
    tangent_x = _unit(eigenvectors[:, 2])
    tangent_y = _unit(np.cross(normal0, tangent_x))
    relative = neighbors - query
    x = relative @ tangent_x
    y = relative @ tangent_y
    z = relative @ normal0
    coeff, fit_rms, condition = _robust_quadratic(x, y, z, cfg.fit_radius_m, cfg)
    normal = _unit(-coeff[1] * tangent_x - coeff[2] * tangent_y + normal0)
    if np.dot(normal, outward_hint) < 0.0:
        normal = -normal
        coeff[1:] *= -1.0
    k1, k2 = _principal_curvatures(coeff)
    # Fitted normal variation over the physical pad footprint.
    footprint = (x * x + y * y) <= cfg.pad_radius_m * cfg.pad_radius_m
    xf, yf = x[footprint], y[footprint]
    dfdx = coeff[1] + coeff[3] * xf + coeff[4] * yf
    dfdy = coeff[2] + coeff[4] * xf + coeff[5] * yf
    local_normals = (-dfdx[:, None] * tangent_x - dfdy[:, None] * tangent_y + normal0)
    local_normals /= np.linalg.norm(local_normals, axis=1, keepdims=True)
    angles = np.rad2deg(np.arccos(np.clip(local_normals @ normal, -1.0, 1.0)))
    boundary_distance, angular_coverage, hole_risk = _boundary_evidence(
        np.column_stack((x, y)), cfg.pad_radius_m)
    count_score = float(np.clip((len(neighbors) - cfg.min_neighbors) / 96.0, 0.0, 1.0))
    rms_score = float(np.exp(-((fit_rms / 0.003) ** 2)))
    condition_score = float(1.0 / (1.0 + max(0.0, condition - 20.0) / 200.0))
    confidence = float(np.clip(0.35 * rms_score + 0.25 * angular_coverage
                               + 0.25 * count_score + 0.15 * condition_score, 0.0, 1.0))
    max_curvature = max(abs(k1), abs(k2))
    return {
        "normal_x": float(normal[0]), "normal_y": float(normal[1]),
        "normal_z": float(normal[2]), "k1_1_m": k1, "k2_1_m": k2,
        "radius1_m": float("inf") if abs(k1) < 1e-9 else 1.0 / abs(k1),
        "radius2_m": float("inf") if abs(k2) < 1e-9 else 1.0 / abs(k2),
        "surface_class": _classification(k1, k2, cfg.flat_curvature_1_m),
        "pad_curvature_ratio": cfg.pad_radius_m * max_curvature,
        "normal_spread_p95_deg": float(np.percentile(angles, 95.0)),
        "normal_spread_max_deg": float(angles.max()),
        "fit_rms_m": fit_rms, "fit_condition": condition,
        "boundary_distance_m": boundary_distance,
        "angular_coverage": angular_coverage, "hole_risk": hole_risk,
        "fit_confidence": confidence, "neighbor_count": int(len(neighbors)),
        "point_density_m2": len(neighbors) / (np.pi * cfg.fit_radius_m ** 2),
    }


def analyse_paths(point_cloud: np.ndarray, scan_dir: str, cfg: GeometryConfig) -> list[dict]:
    tree = cKDTree(point_cloud)
    cloud_center = point_cloud.mean(axis=0)
    rows = []
    for rail in RAILS:
        for segment, path_file in segment_files(scan_dir, rail):
            path = np.load(path_file).astype(np.float64)
            previous_normal = None
            previous_curvature_ratio = None
            previous_query = None
            for waypoint, query in enumerate(path):
                path_step = (0.0 if previous_query is None else
                             float(np.linalg.norm(query - previous_query)))
                continuous = previous_query is None or path_step <= cfg.path_continuity_max_step_m
                nearest_distance, _ = tree.query(query, k=1)
                ids = tree.query_ball_point(query, cfg.support_radius_m)
                if len(ids) > cfg.max_neighbors:
                    distances = np.linalg.norm(point_cloud[ids] - query, axis=1)
                    ids = np.asarray(ids)[np.argsort(distances)[:cfg.max_neighbors]].tolist()
                support = point_cloud[ids]
                fit_mask = np.linalg.norm(support - query, axis=1) <= cfg.fit_radius_m
                fit_points = support[fit_mask]
                fit = fit_local_surface(query, fit_points, cloud_center, cfg)
                normal = np.array([fit["normal_x"], fit["normal_y"], fit["normal_z"]])
                if continuous and previous_normal is not None and np.dot(normal, previous_normal) < 0.0:
                    normal = -normal
                    fit["normal_x"], fit["normal_y"], fit["normal_z"] = normal
                    fit["k1_1_m"], fit["k2_1_m"] = -fit["k2_1_m"], -fit["k1_1_m"]
                    fit["surface_class"] = _classification(
                        fit["k1_1_m"], fit["k2_1_m"], cfg.flat_curvature_1_m)
                normal_angle = (0.0 if previous_normal is None else float(np.rad2deg(
                    np.arccos(np.clip(np.dot(normal, previous_normal), -1.0, 1.0)))))
                deviation_risk = float(np.clip(nearest_distance / cfg.path_deviation_unsafe_m, 0.0, 1.0))
                boundary_risk = float(np.clip(1.0 - fit["boundary_distance_m"] / cfg.pad_radius_m,
                                              0.0, 1.0))
                curvature_jump = (0.0 if previous_curvature_ratio is None or not continuous else
                                  abs(fit["pad_curvature_ratio"] - previous_curvature_ratio))
                curvature_jump_risk = float(np.clip(curvature_jump / 0.15, 0.0, 1.0))
                geometry_risk = max(
                    float(np.clip(fit["pad_curvature_ratio"] / cfg.curvature_unsafe_ratio, 0.0, 1.0)),
                    boundary_risk, fit["hole_risk"], 1.0 - fit["fit_confidence"],
                    deviation_risk, curvature_jump_risk,
                    float(np.clip(fit["normal_spread_p95_deg"] / cfg.normal_spread_unsafe_deg, 0.0, 1.0)),
                )
                unsafe_reasons = []
                if fit["fit_confidence"] < cfg.confidence_unsafe_threshold:
                    unsafe_reasons.append("low_fit_confidence")
                if boundary_risk > cfg.boundary_unsafe_risk:
                    unsafe_reasons.append("point_cloud_boundary")
                if fit["normal_spread_p95_deg"] > cfg.normal_spread_unsafe_deg:
                    unsafe_reasons.append("footprint_normal_spread")
                if fit["pad_curvature_ratio"] > cfg.curvature_unsafe_ratio:
                    unsafe_reasons.append("pad_curvature")
                if nearest_distance > cfg.path_deviation_unsafe_m:
                    unsafe_reasons.append("path_deviation")
                review_reasons = list(unsafe_reasons)
                if fit["fit_confidence"] < cfg.confidence_review_threshold and not unsafe_reasons:
                    review_reasons.append("review_fit_confidence")
                if previous_query is not None and not continuous:
                    review_reasons.append("path_discontinuity")
                rows.append({
                    "rail": rail, "segment": segment, "waypoint": waypoint,
                    "path_file": os.path.basename(path_file),
                    "x_m": float(query[0]), "y_m": float(query[1]), "z_m": float(query[2]),
                    "nearest_cloud_distance_m": float(nearest_distance),
                    "path_step_m": path_step,
                    "path_continuous": continuous,
                    "normal_angle_from_previous_deg": normal_angle,
                    "normal_angle_continuous_deg": normal_angle if continuous else "",
                    **fit,
                    "boundary_risk": boundary_risk,
                    "path_deviation_risk": deviation_risk,
                    "curvature_jump_risk": curvature_jump_risk,
                    "geometry_risk": geometry_risk,
                    "unsafe_geometry_pt_design": bool(unsafe_reasons),
                    "unsafe_reasons": "|".join(unsafe_reasons),
                    "review_reasons": "|".join(review_reasons),
                })
                previous_normal = normal
                previous_curvature_ratio = fit["pad_curvature_ratio"]
                previous_query = query
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> tuple[list[dict], dict]:
    rail_rows = []
    for rail in RAILS:
        selected = [row for row in rows if row["rail"] == rail]
        rail_rows.append({
            "rail": rail, "segments": len(set(row["segment"] for row in selected)),
            "waypoints": len(selected),
            "unsafe_geometry_pt_design": sum(bool(row["unsafe_geometry_pt_design"]) for row in selected),
            "review_required": sum(bool(row["review_reasons"]) for row in selected),
            "fit_confidence_min": min(row["fit_confidence"] for row in selected),
            "fit_confidence_median": float(np.median([row["fit_confidence"] for row in selected])),
            "pad_curvature_ratio_p95": float(np.percentile(
                [row["pad_curvature_ratio"] for row in selected], 95.0)),
            "normal_spread_p95_of_waypoints_deg": float(np.percentile(
                [row["normal_spread_p95_deg"] for row in selected], 95.0)),
            "boundary_risk_p95": float(np.percentile(
                [row["boundary_risk"] for row in selected], 95.0)),
        })
    summary = {
        "waypoints": len(rows),
        "segments": len(set((row["rail"], row["segment"]) for row in rows)),
        "normal_unit_error_max": max(abs(np.linalg.norm([
            row["normal_x"], row["normal_y"], row["normal_z"]]) - 1.0) for row in rows),
        "normal_angle_previous_max_deg": max(row["normal_angle_from_previous_deg"] for row in rows),
        "normal_angle_continuous_max_deg": max(
            row["normal_angle_continuous_deg"] for row in rows
            if row["normal_angle_continuous_deg"] != ""),
        "path_discontinuities": sum(not bool(row["path_continuous"]) for row in rows),
        "fit_rms_p95_m": float(np.percentile([row["fit_rms_m"] for row in rows], 95.0)),
        "fit_confidence_min": min(row["fit_confidence"] for row in rows),
        "fit_confidence_median": float(np.median([row["fit_confidence"] for row in rows])),
        "unsafe_geometry_pt_design": sum(bool(row["unsafe_geometry_pt_design"]) for row in rows),
        "review_required": sum(bool(row["review_reasons"]) for row in rows),
        "surface_class_counts": {name: sum(row["surface_class"] == name for row in rows)
                                 for name in sorted(set(row["surface_class"] for row in rows))},
    }
    return rail_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", default="scan_result/car")
    parser.add_argument("--point-cloud",
                        default="scan_result/car/points/real_camera_surface_points.ply")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if os.path.exists(args.out_dir):
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    cfg = GeometryConfig()
    points = load_ascii_xyz_ply(args.point_cloud)
    rows = analyse_paths(points, args.scan_dir, cfg)
    rail_rows, summary = summarize(rows)
    os.makedirs(args.out_dir)
    write_csv(os.path.join(args.out_dir, "waypoint_geometry.csv"), rows)
    write_csv(os.path.join(args.out_dir, "rail_geometry_summary.csv"), rail_rows)
    finite_fields = ("normal_x", "normal_y", "normal_z", "k1_1_m", "k2_1_m",
                     "pad_curvature_ratio", "normal_spread_p95_deg", "fit_rms_m",
                     "fit_confidence", "boundary_risk", "geometry_risk")
    checks = [
        {"check": "canonical_point_count", "expected": 111968, "actual": len(points),
         "pass": len(points) == 111968},
        {"check": "existing_path_waypoints", "expected": 2498, "actual": len(rows),
         "pass": len(rows) == 2498},
        {"check": "existing_path_segments", "expected": 38, "actual": summary["segments"],
         "pass": summary["segments"] == 38},
        {"check": "unit_normals", "expected": "<=1e-6 max error",
         "actual": summary["normal_unit_error_max"],
         "pass": summary["normal_unit_error_max"] <= 1e-6},
        {"check": "finite_critical_geometry", "expected": True,
         "actual": all(np.isfinite(float(row[field])) for row in rows for field in finite_fields),
         "pass": all(np.isfinite(float(row[field])) for row in rows for field in finite_fields)},
        {"check": "paths_coincident_with_cloud", "expected": "<=1e-5 m max",
         "actual": max(row["nearest_cloud_distance_m"] for row in rows),
         "pass": max(row["nearest_cloud_distance_m"] for row in rows) <= 1e-5},
    ]
    write_csv(os.path.join(args.out_dir, "acceptance_checks.csv"), checks)
    inputs = [args.point_cloud, os.path.join(args.scan_dir, "rail_config.json")]
    inputs.extend(path for rail in RAILS for _, path in segment_files(args.scan_dir, rail))
    write_csv(os.path.join(args.out_dir, "input_checksums.csv"),
              [{"path": os.path.abspath(path), "sha256": sha256(path)} for path in inputs])
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F10B_VEHICLE_GEOMETRY_CPU",
        "status": "ANALYSIS_COMPLETE_THRESHOLDS_NOT_SAFETY_VALIDATED",
        "configuration_pt_design": asdict(cfg),
        "point_cloud": os.path.abspath(args.point_cloud),
        "point_cloud_points": len(points),
        "summary": summary,
        "acceptance_checks_pass": all(bool(check["pass"]) for check in checks),
        "physx_executed": False, "training_performed": False,
        "paths_modified": False, "polishing_v5_modified": False,
        "next_step_allowed": False,
    }
    with open(os.path.join(args.out_dir, "decision.json"), "w", encoding="utf-8") as fh:
        json.dump(decision, fh, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("# Gate F10-B vehicle geometry\n\n"
                 "CPU-only robust quadratic fits at existing C/SL/SR waypoints. "
                 "Risk thresholds are diagnostic PT-DESIGN values and require later validation.\n")
    outputs = sorted(glob.glob(os.path.join(args.out_dir, "*")))
    with open(os.path.join(args.out_dir, "checksums.sha256"), "w", encoding="utf-8") as fh:
        for path in outputs:
            fh.write(f"{sha256(path)}  {os.path.basename(path)}\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
