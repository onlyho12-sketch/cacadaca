#!/usr/bin/env python3
"""Build a seeded multi-surface Local-20 and synthetic quality benchmark."""

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
from gu_proxy import LiteratureGuProxyConfig, relative_gloss_to_gu_proxy


SCENARIOS = (
    {"id": "plane", "kind": "quadratic", "ku": 0.0, "kv": 0.0, "twist": 0.0},
    {"id": "convex_cylinder", "kind": "cylinder", "radius_m": 0.35, "sign": 1.0},
    {"id": "concave_cylinder", "kind": "cylinder", "radius_m": 0.45, "sign": -1.0},
    {"id": "convex_sphere", "kind": "sphere", "radius_m": 0.50, "sign": 1.0},
    {"id": "freeform_mild", "kind": "quadratic", "ku": 2.0, "kv": -0.8, "twist": 0.5},
    {"id": "freeform_strong", "kind": "quadratic", "ku": 4.0, "kv": -2.0, "twist": 1.5},
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--span-m", type=float, default=0.16)
    parser.add_argument("--angle-deg", type=float, default=20.0)
    parser.add_argument("--rig-distance-m", type=float, default=0.42)
    parser.add_argument("--footprint-diameter-m", type=float, default=0.01)
    parser.add_argument("--normal-spread-limit-deg", type=float, default=3.0)
    parser.add_argument("--clearcoat-min-um", type=float, default=40.0)
    parser.add_argument("--clearcoat-max-um", type=float, default=50.0)
    parser.add_argument("--clearcoat-safety-limit-um", type=float, default=35.0)
    parser.add_argument("--target-gu", type=float, default=70.0)
    return parser.parse_args()


def rotated_xy_yaw(vector, yaw_deg):
    angle = math.radians(yaw_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    x, y, z = vector
    return np.asarray([
        cosine * x - sine * y,
        sine * x + cosine * y,
        z,
    ], dtype=float)


def surface(spec, u, v):
    kind = spec["kind"]
    if kind == "cylinder":
        radius = float(spec["radius_m"])
        root = math.sqrt(radius * radius - u * u)
        sign = float(spec["sign"])
        z = sign * (radius - root)
        dz_du, dz_dv = sign * u / root, 0.0
    elif kind == "sphere":
        radius = float(spec["radius_m"])
        root = math.sqrt(radius * radius - u * u - v * v)
        sign = float(spec["sign"])
        z = sign * (radius - root)
        dz_du, dz_dv = sign * u / root, sign * v / root
    elif kind == "quadratic":
        ku, kv, twist = float(spec["ku"]), float(spec["kv"]), float(spec["twist"])
        z = 0.5 * ku * u * u + 0.5 * kv * v * v + twist * u * v
        dz_du, dz_dv = ku * u + twist * v, kv * v + twist * u
    else:
        raise ValueError(f"unknown surface kind: {kind}")
    yaw = float(spec.get("yaw_deg", 0.0))
    point = rotated_xy_yaw([u, v, z], yaw)
    normal = normalize(rotated_xy_yaw([-dz_du, -dz_dv, 1.0], yaw))
    tangent = normalize(rotated_xy_yaw([1.0, 0.0, dz_du], yaw))
    return point, normal, tangent


def angle_deg(a, b):
    cosine = float(np.clip(np.dot(normalize(a), normalize(b)), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def footprint_spread(spec, u, v, diameter_m):
    center = surface(spec, u, v)[1]
    radius = diameter_m / 2.0
    offsets = np.linspace(-radius, radius, 5)
    angles = []
    for du in offsets:
        for dv in offsets:
            if du * du + dv * dv <= radius * radius + 1.0e-15:
                angles.append(angle_deg(center, surface(spec, u + du, v + dv)[1]))
    return float(np.percentile(angles, 95))


def resolved_scenarios(seed):
    rng = np.random.default_rng(seed)
    resolved = []
    for index, template in enumerate(SCENARIOS):
        spec = dict(template)
        spec["scenario_index"] = index
        spec["scenario_seed"] = int(seed + index * 1009)
        spec["yaw_deg"] = float(rng.uniform(-75.0, 75.0))
        resolved.append(spec)
    return resolved


def generate_rows(
    seed=20260828, grid=5, span_m=0.16, angle=20.0,
    rig_distance=0.42, footprint=0.01, spread_limit=3.0,
    clearcoat_min=40.0, clearcoat_max=50.0,
    clearcoat_safety=35.0, target_gu=70.0,
):
    if grid < 3 or grid % 2 == 0:
        raise ValueError("grid must be an odd integer >= 3")
    if not 0.0 < clearcoat_min < clearcoat_max:
        raise ValueError("clearcoat range must satisfy 0 < min < max")
    coordinates = np.linspace(-span_m / 2.0, span_m / 2.0, grid)
    config = LiteratureGuProxyConfig(target_gu=target_gu)
    rows = []
    specs = resolved_scenarios(seed)
    for spec in specs:
        rng = np.random.default_rng(spec["scenario_seed"])
        severity = rng.uniform(0.25, 1.0, size=(grid, grid))
        pristine = (int(rng.integers(0, grid)), int(rng.integers(0, grid)))
        severity[pristine] = 0.0
        for row_index, v in enumerate(coordinates, start=1):
            for column_index, u in enumerate(coordinates, start=1):
                point, normal, tangent = surface(spec, float(u), float(v))
                rig = measurement_pose(
                    point, normal, angle, rig_distance, tangent_hint=tangent
                )
                incoming = normalize(point - rig["light_position"])
                reflected = normalize(
                    incoming - 2.0 * np.dot(incoming, normal) * normal
                )
                detector = normalize(rig["camera_position"] - point)
                cell_severity = float(severity[row_index - 1, column_index - 1])
                roughness_before = 0.10 + 0.25 * cell_severity
                roughness_after = 0.10 + 0.02 * cell_severity
                scratch_before = cell_severity
                scratch_after = 0.08 * cell_severity
                clearcoat_before = float(rng.uniform(clearcoat_min, clearcoat_max))
                removed = 0.0 if cell_severity == 0.0 else float(
                    rng.uniform(1.0, 3.2) * (0.65 + 0.35 * cell_severity)
                )
                clearcoat_after = clearcoat_before - removed
                relative_before = float(np.clip(1.0 - 0.78 * cell_severity, 0.0, 1.0))
                relative_after = float(np.clip(1.0 - 0.08 * cell_severity, 0.0, 1.0))
                gu_before = relative_gloss_to_gu_proxy(relative_before, config)
                gu_after = relative_gloss_to_gu_proxy(relative_after, config)
                spread = footprint_spread(spec, float(u), float(v), footprint)
                incident = angle_deg(rig["light_direction_from_surface"], normal)
                detection = angle_deg(rig["detector_direction_from_surface"], normal)
                reflection_error = angle_deg(reflected, detector)
                rows.append({
                    "data_origin": "synthetic_surface_generalization_not_rl",
                    "scenario_id": spec["id"],
                    "scenario_seed": spec["scenario_seed"],
                    "surface_kind": spec["kind"],
                    "surface_yaw_deg": spec["yaw_deg"],
                    "grid_row": row_index,
                    "grid_column": column_index,
                    "position_x_m": float(point[0]),
                    "position_y_m": float(point[1]),
                    "position_z_m": float(point[2]),
                    "normal_x": float(normal[0]),
                    "normal_y": float(normal[1]),
                    "normal_z": float(normal[2]),
                    "incident_angle_deg": incident,
                    "detection_angle_deg": detection,
                    "reflection_error_deg": reflection_error,
                    "normal_spread_p95_deg": spread,
                    "footprint_valid": spread <= spread_limit,
                    "defect_severity": cell_severity,
                    "roughness_before": roughness_before,
                    "roughness_after": roughness_after,
                    "scratch_before": scratch_before,
                    "scratch_after": scratch_after,
                    "clearcoat_before_um": clearcoat_before,
                    "clearcoat_removed_um": removed,
                    "clearcoat_after_um": clearcoat_after,
                    "relative_gloss_before_not_gu": relative_before,
                    "relative_gloss_after_not_gu": relative_after,
                    "gu_proxy_before": gu_before,
                    "gu_proxy_after": gu_after,
                    "gu_proxy_pass": gu_after >= target_gu,
                    "clearcoat_safety_pass": clearcoat_after >= clearcoat_safety,
                })
    return specs, rows


def save_plot(path, specs, rows, grid):
    figure, axes = plt.subplots(3, len(specs), figsize=(19.0, 9.0), constrained_layout=True)
    panels = (
        ("normal_spread_p95_deg", "Normal spread p95 (deg)", "viridis"),
        ("gu_proxy_after", "20 deg GU proxy after", "viridis"),
        ("clearcoat_after_um", "Clearcoat after (um)", "plasma"),
    )
    for column, spec in enumerate(specs):
        scenario_rows = [row for row in rows if row["scenario_id"] == spec["id"]]
        for panel_row, (key, title, cmap) in enumerate(panels):
            data = np.asarray([row[key] for row in scenario_rows]).reshape(grid, grid)
            axis = axes[panel_row, column]
            image = axis.imshow(data, origin="lower", cmap=cmap)
            axis.set_title(f"{spec['id']}\n{title}", fontsize=9)
            axis.set_xticks(range(grid), range(1, grid + 1), fontsize=6)
            axis.set_yticks(range(grid), range(1, grid + 1), fontsize=6)
            for row in range(grid):
                for cell_column in range(grid):
                    axis.text(
                        cell_column, row, f"{data[row, cell_column]:.1f}",
                        ha="center", va="center", fontsize=5,
                    )
            figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
    figure.suptitle("Seeded surface-generalization benchmark (synthetic, not RL)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(output_dir, **kwargs):
    specs, rows = generate_rows(**kwargs)
    target_angle = float(kwargs.get("angle", 20.0))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "surface_generalization_cells.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scenario_summaries = []
    for spec in specs:
        selected = [row for row in rows if row["scenario_id"] == spec["id"]]
        scenario_summaries.append({
            "scenario_id": spec["id"],
            "scenario_seed": spec["scenario_seed"],
            "surface_definition": spec,
            "sample_count": len(selected),
            "max_incident_angle_error_deg": max(
                abs(row["incident_angle_deg"] - target_angle) for row in selected
            ),
            "max_detection_angle_error_deg": max(
                abs(row["detection_angle_deg"] - target_angle) for row in selected
            ),
            "max_reflection_error_deg": max(row["reflection_error_deg"] for row in selected),
            "footprint_pass_count": sum(row["footprint_valid"] for row in selected),
            "gu_proxy_pass_count": sum(row["gu_proxy_pass"] for row in selected),
            "clearcoat_safety_pass_count": sum(
                row["clearcoat_safety_pass"] for row in selected
            ),
            "minimum_clearcoat_after_um": min(
                row["clearcoat_after_um"] for row in selected
            ),
            "rtx_measurement_performed": False,
        })
    summary = {
        "status": "surface_generalization_geometry_and_synthetic_quality_prepared",
        "data_origin": "synthetic_surface_generalization_not_rl",
        "seed": kwargs.get("seed", 20260828),
        "target_angle_deg": target_angle,
        "scenario_count": len(specs),
        "sample_count": len(rows),
        "scenarios": scenario_summaries,
        "geometry_passed": all(
            item["max_incident_angle_error_deg"] < 1.0e-6
            and item["max_detection_angle_error_deg"] < 1.0e-6
            and item["max_reflection_error_deg"] < 1.0e-6
            for item in scenario_summaries
        ),
        "footprint_all_passed": all(
            item["footprint_pass_count"] == item["sample_count"]
            for item in scenario_summaries
        ),
        "synthetic_quality_all_passed": all(
            item["gu_proxy_pass_count"] == item["sample_count"]
            and item["clearcoat_safety_pass_count"] == item["sample_count"]
            for item in scenario_summaries
        ),
        "important": (
            "All quality/clearcoat values are seeded synthetic design data, not RL "
            "output or physical measurements. RTX status is reported separately."
        ),
    }
    json_path = output_dir / "surface_generalization_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_path = output_dir / "surface_generalization_heatmaps.png"
    save_plot(plot_path, specs, rows, int(math.sqrt(len(rows) / len(specs))))
    return csv_path, json_path, plot_path, summary


def main():
    args = parse_args()
    csv_path, json_path, plot_path, summary = run(
        args.output_dir,
        seed=args.seed,
        grid=args.grid,
        span_m=args.span_m,
        angle=args.angle_deg,
        rig_distance=args.rig_distance_m,
        footprint=args.footprint_diameter_m,
        spread_limit=args.normal_spread_limit_deg,
        clearcoat_min=args.clearcoat_min_um,
        clearcoat_max=args.clearcoat_max_um,
        clearcoat_safety=args.clearcoat_safety_limit_um,
        target_gu=args.target_gu,
    )
    print("")
    print("=" * 80)
    print("다중 표면 일반화 벤치마크 — 기하 및 합성 품질 상태")
    print(f"시나리오/측정점 : {summary['scenario_count']}개 / {summary['sample_count']}개")
    for item in summary["scenarios"]:
        print(
            f"  {item['scenario_id']:<18} Local20=PASS, "
            f"footprint={item['footprint_pass_count']}/{item['sample_count']}, "
            f"GU={item['gu_proxy_pass_count']}/{item['sample_count']}, "
            f"clearcoat={item['clearcoat_safety_pass_count']}/{item['sample_count']}, "
            "RTX=NOT_RUN"
        )
    print(f"기하 최종판정       : {'통과' if summary['geometry_passed'] else '실패'}")
    print(f"CSV                : {csv_path}")
    print(f"JSON               : {json_path}")
    print(f"히트맵             : {plot_path}")
    print("=" * 80)
    if not summary["geometry_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
