#!/usr/bin/env python3
"""Validate Local 20-degree geometry on multiple BMW body regions."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT))

from config import gloss_config as cfg
from config.vehicle_region_profiles import VEHICLE_REGION_PROFILES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument(
        "--target-prim-suffix",
        default="bmw_z4_car_007_color_polySurface37",
    )
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--footprint-diameter-m", type=float, default=0.01)
    parser.add_argument("--normal-spread-limit-deg", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.asset.is_file():
        parser.error(f"--asset not found: {args.asset}")
    if args.grid < 3 or args.grid % 2 == 0:
        parser.error("--grid must be an odd integer >= 3")
    return args


args = parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})


def find_unique_mesh(stage, UsdGeom, suffix):
    matches = [
        prim for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh) and str(prim.GetPath()).endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one mesh ending with {suffix!r}, found {len(matches)}")
    return matches[0]


def angle_deg(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    cosine = float(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))
    return math.degrees(math.acos(cosine))


def save_heatmap(path, rows, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, (region_id, profile) in zip(axes.flat, VEHICLE_REGION_PROFILES.items()):
        region_rows = [row for row in rows if row["region_id"] == region_id]
        values = np.asarray([
            row["footprint_normal_spread_p95_deg"] for row in region_rows
        ]).reshape(grid, grid)
        image = axis.imshow(values, origin="lower", cmap="viridis", vmin=0.0)
        for y in range(grid):
            for x in range(grid):
                axis.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center", fontsize=7)
        axis.set_title(f"{region_id}\n10 mm footprint normal spread (deg)")
        axis.set_xticks(range(grid), range(1, grid + 1))
        axis.set_yticks(range(grid), range(1, grid + 1))
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("BMW Z4 multi-region Local 20° geometry validation")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    from pxr import Gf, Usd, UsdGeom

    from scripts.gloss_geometry import measurement_pose, normalize
    from scripts.mesh_surface_sampling import sample_planar_grid, triangulate_faces

    stage = Usd.Stage.Open(str(args.asset.resolve()))
    stage.Load()
    target = find_unique_mesh(stage, UsdGeom, args.target_prim_suffix)
    mesh = UsdGeom.Mesh(target)
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(target)
    points = np.asarray([
        tuple(transform.Transform(Gf.Vec3d(*[float(value) for value in point])))
        for point in (mesh.GetPointsAttr().Get() or [])
    ], dtype=float)
    triangles = triangulate_faces(
        mesh.GetFaceVertexCountsAttr().Get() or [],
        mesh.GetFaceVertexIndicesAttr().Get() or [],
    )

    rows = []
    region_summaries = []
    for region_id, profile in VEHICLE_REGION_PROFILES.items():
        sampled = sample_planar_grid(
            points, triangles, profile["center_m"], profile["axis_u"],
            profile["axis_v"], profile["ray_direction"], profile["span_m"],
            args.grid,
        )
        hit_count = sum(item["hit"] is not None for item in sampled)
        if hit_count != args.grid * args.grid:
            missing = [
                [item["grid_row"], item["grid_column"]]
                for item in sampled if item["hit"] is None
            ]
            region_summaries.append({
                "region_id": region_id,
                "label": profile["label"],
                "mesh_hit_count": hit_count,
                "sample_count": args.grid * args.grid,
                "missing_cells": missing,
                "passed": False,
            })
            continue

        region_rows = []
        for item in sampled:
            hit = item["hit"]
            point = hit["point"]
            normal = hit["normal"]
            rig = measurement_pose(
                point, normal, cfg.INCIDENT_ANGLE_DEG, cfg.RIG_DISTANCE_M,
                tangent_hint=profile["axis_u"],
            )
            incoming = normalize(point - rig["light_position"])
            reflected = normalize(incoming - 2.0 * np.dot(incoming, normal) * normal)
            detector = normalize(rig["camera_position"] - point)
            footprint = sample_planar_grid(
                points, triangles, item["plane_point"], profile["axis_u"],
                profile["axis_v"], profile["ray_direction"],
                args.footprint_diameter_m, 5,
            )
            footprint_normals = [
                neighbor["hit"]["normal"]
                for neighbor in footprint if neighbor["hit"] is not None
            ]
            spreads = [angle_deg(normal, neighbor) for neighbor in footprint_normals]
            spread_p95 = float(np.percentile(spreads, 95)) if spreads else float("inf")
            row = {
                "region_id": region_id,
                "region_label": profile["label"],
                "grid_row": item["grid_row"],
                "grid_column": item["grid_column"],
                "position_x_m": float(point[0]),
                "position_y_m": float(point[1]),
                "position_z_m": float(point[2]),
                "normal_x": float(normal[0]),
                "normal_y": float(normal[1]),
                "normal_z": float(normal[2]),
                "incident_angle_deg": angle_deg(rig["light_direction_from_surface"], normal),
                "detection_angle_deg": angle_deg(rig["detector_direction_from_surface"], normal),
                "specular_reflection_error_deg": angle_deg(reflected, detector),
                "footprint_mesh_hit_count": len(footprint_normals),
                "footprint_normal_spread_p95_deg": spread_p95,
                "footprint_valid": (
                    len(footprint_normals) == 25
                    and spread_p95 <= args.normal_spread_limit_deg
                ),
            }
            rows.append(row)
            region_rows.append(row)

        max_incident_error = max(
            abs(row["incident_angle_deg"] - cfg.INCIDENT_ANGLE_DEG)
            for row in region_rows
        )
        max_detection_error = max(
            abs(row["detection_angle_deg"] - cfg.INCIDENT_ANGLE_DEG)
            for row in region_rows
        )
        max_reflection_error = max(
            row["specular_reflection_error_deg"] for row in region_rows
        )
        footprint_valid_count = sum(row["footprint_valid"] for row in region_rows)
        region_summaries.append({
            "region_id": region_id,
            "label": profile["label"],
            "mesh_hit_count": hit_count,
            "sample_count": len(region_rows),
            "footprint_valid_count": footprint_valid_count,
            "max_footprint_normal_spread_p95_deg": max(
                row["footprint_normal_spread_p95_deg"] for row in region_rows
            ),
            "max_incident_angle_error_deg": max_incident_error,
            "max_detection_angle_error_deg": max_detection_error,
            "max_specular_reflection_error_deg": max_reflection_error,
            "passed": (
                hit_count == args.grid * args.grid
                and footprint_valid_count == args.grid * args.grid
                and max_incident_error < 1.0e-6
                and max_detection_error < 1.0e-6
                and max_reflection_error < 1.0e-6
            ),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "vehicle_multi_region_local_20.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "mode": "actual_bmw_mesh_multi_region_local_20_geometry",
        "asset": str(args.asset.resolve()),
        "target_prim": str(target.GetPath()),
        "mesh_bounds_min_m": points.min(axis=0).tolist(),
        "mesh_bounds_max_m": points.max(axis=0).tolist(),
        "region_count": len(VEHICLE_REGION_PROFILES),
        "region_passed_count": sum(item["passed"] for item in region_summaries),
        "sample_count": len(rows),
        "target_angle_deg": cfg.INCIDENT_ANGLE_DEG,
        "footprint_diameter_m": args.footprint_diameter_m,
        "normal_spread_limit_deg": args.normal_spread_limit_deg,
        "regions": region_summaries,
        "is_rtx_measurement": False,
        "passed": all(item["passed"] for item in region_summaries),
        "important": "Geometry only. No HDR intensity, relative gloss, GU proxy, or actual GU is produced.",
    }
    summary_path = args.output_dir / "vehicle_multi_region_local_20_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    heatmap_path = args.output_dir / "vehicle_multi_region_footprint_heatmaps.png"
    if len(rows) == len(VEHICLE_REGION_PROFILES) * args.grid * args.grid:
        save_heatmap(heatmap_path, rows, args.grid)

    print("")
    print("=" * 80)
    print("BMW Z4 다중 영역 실제 Mesh Local 20° 기하 검증")
    print("=" * 80)
    for item in region_summaries:
        print(
            f"{item['label']:<18} Mesh {item['mesh_hit_count']:>2}/25, "
            f"footprint {item.get('footprint_valid_count', 0):>2}/25, "
            f"판정={'통과' if item['passed'] else '실패'}"
        )
    print(f"전체 판정: {summary['region_passed_count']}/{summary['region_count']} 통과")
    print("주의: 이 단계는 기하 검증이며 RTX 광학 측정이나 GU가 아님")
    print(f"CSV: {csv_path}")
    print(f"JSON: {summary_path}")
    print(f"히트맵: {heatmap_path}")
    if not summary["passed"]:
        raise RuntimeError(f"Vehicle multi-region geometry failed; see {summary_path}")


try:
    main()
finally:
    simulation_app.close()
