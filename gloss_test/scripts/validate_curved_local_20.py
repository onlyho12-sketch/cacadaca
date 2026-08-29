#!/usr/bin/env python3
"""Validate local 20-degree source/camera poses on analytic curved patches."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gloss_geometry import measurement_pose, normalize


PROFILES = ("cylinder", "sphere", "freeform")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--span-m", type=float, default=0.16)
    parser.add_argument("--angle-deg", type=float, default=20.0)
    parser.add_argument("--distance-m", type=float, default=0.42)
    parser.add_argument("--footprint-diameter-m", type=float, default=0.01)
    parser.add_argument("--normal-spread-limit-deg", type=float, default=3.0)
    return parser.parse_args()


def surface(profile, u, v):
    """Return point, outward normal and +u path tangent for a smooth patch."""
    if profile == "plane":
        z = 0.0
        dz_du, dz_dv = 0.0, 0.0
    elif profile == "cylinder":
        radius = 0.35
        root = math.sqrt(radius ** 2 - u ** 2)
        z = radius - root
        dz_du, dz_dv = u / root, 0.0
    elif profile == "concave_cylinder":
        radius = 0.45
        root = math.sqrt(radius ** 2 - u ** 2)
        z = -(radius - root)
        dz_du, dz_dv = -u / root, 0.0
    elif profile == "sphere":
        radius = 0.50
        root = math.sqrt(radius ** 2 - u ** 2 - v ** 2)
        z = radius - root
        dz_du, dz_dv = u / root, v / root
    elif profile == "freeform":
        curvature_u, curvature_v, twist = 3.0, -1.2, 1.0
        z = 0.5 * curvature_u * u ** 2 + 0.5 * curvature_v * v ** 2 + twist * u * v
        dz_du = curvature_u * u + twist * v
        dz_dv = curvature_v * v + twist * u
    elif profile == "freeform_strong":
        curvature_u, curvature_v, twist = 4.0, -2.0, 1.5
        z = 0.5 * curvature_u * u ** 2 + 0.5 * curvature_v * v ** 2 + twist * u * v
        dz_du = curvature_u * u + twist * v
        dz_dv = curvature_v * v + twist * u
    else:
        raise ValueError(f"unknown profile: {profile}")
    point = np.array([u, v, z], dtype=float)
    tangent_u = normalize([1.0, 0.0, dz_du])
    normal = normalize([-dz_du, -dz_dv, 1.0])
    return point, normal, tangent_u


def angle_deg(a, b):
    cosine = float(np.clip(np.dot(normalize(a), normalize(b)), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def normal_spread(profile, u, v, footprint_diameter_m):
    center_normal = surface(profile, u, v)[1]
    radius = footprint_diameter_m / 2.0
    offsets = np.linspace(-radius, radius, 5)
    angles = []
    for du in offsets:
        for dv in offsets:
            if du ** 2 + dv ** 2 <= radius ** 2 + 1e-15:
                neighbor_normal = surface(profile, u + du, v + dv)[1]
                angles.append(angle_deg(center_normal, neighbor_normal))
    return float(np.percentile(angles, 95))


def validate_profile(profile, grid, span_m, angle, distance, footprint, spread_limit):
    coordinates = np.linspace(-span_m / 2.0, span_m / 2.0, grid)
    rows = []
    for row_index, v in enumerate(coordinates, start=1):
        for column_index, u in enumerate(coordinates, start=1):
            point, normal, path_tangent = surface(profile, float(u), float(v))
            rig = measurement_pose(
                point, normal, angle, distance, tangent_hint=path_tangent
            )
            incident_angle = angle_deg(rig["light_direction_from_surface"], normal)
            detection_angle = angle_deg(rig["detector_direction_from_surface"], normal)
            incoming = normalize(point - rig["light_position"])
            reflected = normalize(incoming - 2.0 * np.dot(incoming, normal) * normal)
            detector_direction = normalize(rig["camera_position"] - point)
            reflection_error = angle_deg(reflected, detector_direction)
            spread = normal_spread(profile, float(u), float(v), footprint)
            rows.append({
                "profile": profile,
                "grid_row": row_index,
                "grid_column": column_index,
                "u_m": float(u),
                "v_m": float(v),
                "position_x_m": float(point[0]),
                "position_y_m": float(point[1]),
                "position_z_m": float(point[2]),
                "normal_x": float(normal[0]),
                "normal_y": float(normal[1]),
                "normal_z": float(normal[2]),
                "tangent_x": float(rig["tangent"][0]),
                "tangent_y": float(rig["tangent"][1]),
                "tangent_z": float(rig["tangent"][2]),
                "incident_angle_deg": incident_angle,
                "detection_angle_deg": detection_angle,
                "specular_reflection_error_deg": reflection_error,
                "normal_spread_p95_deg": spread,
                "footprint_valid": spread <= spread_limit,
            })
    return rows


def save_plot(path, rows, grid):
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for axis, profile in zip(axes, PROFILES):
        profile_rows = [row for row in rows if row["profile"] == profile]
        spread = np.asarray([row["normal_spread_p95_deg"] for row in profile_rows]).reshape(grid, grid)
        image = axis.imshow(spread, origin="lower", cmap="viridis")
        axis.set_title(f"{profile}\nfootprint normal spread (p95)")
        axis.set_xlabel("grid column")
        axis.set_ylabel("grid row")
        axis.set_xticks(range(grid), range(1, grid + 1))
        axis.set_yticks(range(grid), range(1, grid + 1))
        for y in range(grid):
            for x in range(grid):
                axis.text(x, y, f"{spread[y, x]:.2f}°", ha="center", va="center", fontsize=7)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Curved-surface Local 20° geometry validation")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(output_dir, grid=5, span_m=0.16, angle=20.0, distance=0.42,
        footprint=0.01, spread_limit=3.0):
    if grid < 3 or grid % 2 == 0:
        raise ValueError("grid must be an odd integer >= 3")
    rows = []
    for profile in PROFILES:
        rows.extend(validate_profile(
            profile, grid, span_m, angle, distance, footprint, spread_limit
        ))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "curved_local_20_poses.csv"
    json_path = output_dir / "curved_local_20_summary.json"
    plot_path = output_dir / "curved_local_20_normal_spread.png"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    max_incident_error = max(abs(row["incident_angle_deg"] - angle) for row in rows)
    max_detection_error = max(abs(row["detection_angle_deg"] - angle) for row in rows)
    max_reflection_error = max(row["specular_reflection_error_deg"] for row in rows)
    invalid_count = sum(not row["footprint_valid"] for row in rows)
    summary = {
        "mode": "analytic_curved_patch_local_20_geometry_validation",
        "profiles": list(PROFILES),
        "grid_per_profile": [grid, grid],
        "sample_count": len(rows),
        "target_angle_deg": angle,
        "rig_distance_m": distance,
        "footprint_diameter_m": footprint,
        "normal_spread_limit_deg": spread_limit,
        "max_incident_angle_error_deg": max_incident_error,
        "max_detection_angle_error_deg": max_detection_error,
        "max_specular_reflection_error_deg": max_reflection_error,
        "curvature_invalid_sample_count": invalid_count,
        "geometry_passed": (
            max_incident_error < 1e-6
            and max_detection_error < 1e-6
            and max_reflection_error < 1e-6
        ),
        "is_rtx_measurement": False,
        "important": "This validates poses only; it does not produce optical intensity or GU.",
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_plot(plot_path, rows, grid)
    return csv_path, json_path, plot_path, summary


def print_terminal_report(rows, summary):
    print("")
    print("=" * 72)
    print("곡면 Local 20° 측정기하 검증 결과")
    print("=" * 72)
    print("검사 내용")
    print("  1. 광원 방향이 각 측정점의 국소 표면 법선에서 20°인지")
    print("  2. 카메라 방향이 반대편에서 국소 표면 법선 기준 20°인지")
    print("  3. 광원에서 들어온 빛의 정반사 방향과 카메라 방향이 일치하는지")
    print("  4. 10 mm 측정 footprint 안의 법선 변화가 3° 이하인지")
    print("")
    for profile in PROFILES:
        profile_rows = [row for row in rows if row["profile"] == profile]
        incident = [row["incident_angle_deg"] for row in profile_rows]
        detection = [row["detection_angle_deg"] for row in profile_rows]
        reflection = [row["specular_reflection_error_deg"] for row in profile_rows]
        spread = [row["normal_spread_p95_deg"] for row in profile_rows]
        valid_count = sum(row["footprint_valid"] for row in profile_rows)
        print(f"[{profile}] 측정점 {len(profile_rows)}개")
        print(f"  실제 입사각 범위       : {min(incident):.9f}° ~ {max(incident):.9f}°")
        print(f"  실제 검출각 범위       : {min(detection):.9f}° ~ {max(detection):.9f}°")
        print(f"  최대 정반사 방향 오차 : {max(reflection):.9f}°")
        print(f"  footprint 법선 변화   : {min(spread):.3f}° ~ {max(spread):.3f}°")
        print(f"  footprint 판정        : {valid_count}/{len(profile_rows)} 통과")
        print("")
    print(f"전체 측정 자세          : {summary['sample_count']}개")
    print(f"Local 20° 기하 최종판정 : {'통과' if summary['geometry_passed'] else '실패'}")
    print("광학 세기·GU 측정 여부  : 미측정 (현재 단계는 광원/카메라 자세 검증)")
    print("=" * 72)


def main():
    args = parse_args()
    csv_path, json_path, plot_path, summary = run(
        args.output_dir, args.grid, args.span_m, args.angle_deg, args.distance_m,
        args.footprint_diameter_m, args.normal_spread_limit_deg,
    )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        report_rows = []
        for row in csv.DictReader(handle):
            report_rows.append({
                "profile": row["profile"],
                "incident_angle_deg": float(row["incident_angle_deg"]),
                "detection_angle_deg": float(row["detection_angle_deg"]),
                "specular_reflection_error_deg": float(row["specular_reflection_error_deg"]),
                "normal_spread_p95_deg": float(row["normal_spread_p95_deg"]),
                "footprint_valid": row["footprint_valid"].lower() == "true",
            })
    print_terminal_report(report_rows, summary)
    print(f"[Curved Local 20] CSV: {csv_path}")
    print(f"[Curved Local 20] JSON: {json_path}")
    print(f"[Curved Local 20] plot: {plot_path}")
    if not summary["geometry_passed"]:
        raise RuntimeError(f"Local 20-degree geometry failed; see {json_path}")


if __name__ == "__main__":
    main()
