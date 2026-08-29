#!/usr/bin/env python3
"""Generate repeatable multi-seed BMW polishing states and representative RTX plans."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gu_proxy import LiteratureGuProxyConfig, relative_gloss_to_gu_proxy


REGION_ORDER = (
    "hood",
    "roof",
    "negative_x_door",
    "positive_x_door",
    "negative_x_front_fender",
    "positive_x_front_fender",
)


def parse_seeds(value):
    seeds = []
    for token in value.split(","):
        seed = int(token.strip())
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        default="20260828,20260829,20260830,20260831,20260832",
    )
    parser.add_argument("--clearcoat-min-um", type=float, default=40.0)
    parser.add_argument("--clearcoat-max-um", type=float, default=50.0)
    parser.add_argument("--clearcoat-safety-limit-um", type=float, default=35.0)
    parser.add_argument("--target-gu-proxy", type=float, default=70.0)
    args = parser.parse_args()
    try:
        args.seeds_parsed = parse_seeds(args.seeds)
    except ValueError as error:
        parser.error(str(error))
    if not args.geometry_csv.is_file():
        parser.error(f"geometry CSV not found: {args.geometry_csv}")
    if not 0.0 < args.clearcoat_min_um < args.clearcoat_max_um:
        parser.error("clearcoat range must satisfy 0 < min < max")
    if args.clearcoat_safety_limit_um <= 0.0:
        parser.error("clearcoat safety limit must be positive")
    return args


def read_geometry(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keyed = {}
    for row in rows:
        key = (row["region_id"], int(row["grid_row"]), int(row["grid_column"]))
        if key in keyed:
            raise ValueError(f"duplicate geometry cell: {key}")
        keyed[key] = row
    expected = {
        (region, grid_row, grid_column)
        for region in REGION_ORDER
        for grid_row in range(1, 6)
        for grid_column in range(1, 6)
    }
    if set(keyed) != expected:
        raise ValueError(
            "geometry must contain six complete 5x5 regions; "
            f"missing={sorted(expected - set(keyed))}, "
            f"extra={sorted(set(keyed) - expected)}"
        )
    return keyed


def _region_seed(base_seed, region_index):
    return int(base_seed + region_index * 1009)


def generate_rows(
    geometry,
    seeds,
    clearcoat_min=40.0,
    clearcoat_max=50.0,
    clearcoat_safety=35.0,
    target_gu=70.0,
):
    config = LiteratureGuProxyConfig(target_gu=target_gu)
    rows = []
    representative = []
    state_maps = {}
    for seed in seeds:
        for region_index, region_id in enumerate(REGION_ORDER):
            scenario_seed = _region_seed(seed, region_index)
            rng = np.random.default_rng(scenario_seed)
            severity = rng.uniform(0.25, 1.0, size=(5, 5)).astype(np.float32)
            pristine = (int(rng.integers(0, 5)), int(rng.integers(0, 5)))
            severity[pristine] = 0.0
            roughness_before = 0.10 + 0.25 * severity
            roughness_after = 0.10 + 0.02 * severity
            scratch_before = severity.copy()
            scratch_after = 0.08 * severity
            clearcoat_before = rng.uniform(clearcoat_min, clearcoat_max, size=(5, 5))
            removal_scale = rng.uniform(1.0, 3.2, size=(5, 5))
            clearcoat_removed = removal_scale * (0.65 + 0.35 * severity)
            clearcoat_removed[pristine] = 0.0
            clearcoat_after = clearcoat_before - clearcoat_removed
            relative_before = np.clip(1.0 - 0.78 * severity, 0.0, 1.0)
            relative_after = np.clip(1.0 - 0.08 * severity, 0.0, 1.0)
            gu_before = relative_gloss_to_gu_proxy(relative_before, config)
            gu_after = relative_gloss_to_gu_proxy(relative_after, config)

            state_maps[(seed, region_id)] = {
                "roughness_before": roughness_before.astype(np.float32),
                "roughness_after": roughness_after.astype(np.float32),
                "scratch_before": scratch_before.astype(np.float32),
                "scratch_after": scratch_after.astype(np.float32),
                "clearcoat_before_um": clearcoat_before.astype(np.float32),
                "clearcoat_after_um": clearcoat_after.astype(np.float32),
                "clearcoat_removed_um": clearcoat_removed.astype(np.float32),
                "gu_proxy_before": np.asarray(gu_before, dtype=np.float32),
                "gu_proxy_after": np.asarray(gu_after, dtype=np.float32),
                "severity": severity,
            }

            positive = severity[severity > 0.0]
            median_value = float(np.median(positive))
            choices = {
                "maximum_defect": tuple(np.unravel_index(np.argmax(severity), severity.shape)),
                "median_defect": tuple(
                    np.unravel_index(
                        np.argmin(np.where(severity > 0.0, np.abs(severity - median_value), np.inf)),
                        severity.shape,
                    )
                ),
                "pristine": pristine,
                "center_reference": (2, 2),
            }
            used = set()
            for reason, zero_cell in choices.items():
                cell = (zero_cell[0] + 1, zero_cell[1] + 1)
                if cell in used:
                    continue
                used.add(cell)
                representative.append({
                    "seed": seed,
                    "region_id": region_id,
                    "scenario_seed": scenario_seed,
                    "grid_row": cell[0],
                    "grid_column": cell[1],
                    "selection_reason": reason,
                    "defect_severity": float(severity[zero_cell]),
                })

            for grid_row in range(1, 6):
                for grid_column in range(1, 6):
                    key = (region_id, grid_row, grid_column)
                    source = geometry[key]
                    index = (grid_row - 1, grid_column - 1)
                    gu_pass = float(gu_after[index]) >= target_gu
                    clearcoat_pass = float(clearcoat_after[index]) >= clearcoat_safety
                    rows.append({
                        "data_origin": "synthetic_vehicle_repeatability_not_rl",
                        "seed": seed,
                        "scenario_seed": scenario_seed,
                        "region_id": region_id,
                        "grid_row": grid_row,
                        "grid_column": grid_column,
                        "position_x_m": float(source["position_x_m"]),
                        "position_y_m": float(source["position_y_m"]),
                        "position_z_m": float(source["position_z_m"]),
                        "normal_x": float(source["normal_x"]),
                        "normal_y": float(source["normal_y"]),
                        "normal_z": float(source["normal_z"]),
                        "defect_severity": float(severity[index]),
                        "roughness_before": float(roughness_before[index]),
                        "roughness_after": float(roughness_after[index]),
                        "scratch_before": float(scratch_before[index]),
                        "scratch_after": float(scratch_after[index]),
                        "ra_before_um": float(0.0805 + 0.35 * severity[index]),
                        "ra_after_um": float(0.0805 + 0.02 * severity[index]),
                        "rz_before_um": float(0.45 + 1.8 * severity[index]),
                        "rz_after_um": float(0.45 + 0.12 * severity[index]),
                        "clearcoat_before_um": float(clearcoat_before[index]),
                        "clearcoat_removed_um": float(clearcoat_removed[index]),
                        "clearcoat_after_um": float(clearcoat_after[index]),
                        "relative_gloss_before_not_gu": float(relative_before[index]),
                        "relative_gloss_after_not_gu": float(relative_after[index]),
                        "gu_proxy_before": float(gu_before[index]),
                        "gu_proxy_after": float(gu_after[index]),
                        "gu_proxy_improvement": float(gu_after[index] - gu_before[index]),
                        "gu_proxy_target_pass": bool(gu_pass),
                        "clearcoat_safety_pass": bool(clearcoat_pass),
                        "all_targets_pass": bool(gu_pass and clearcoat_pass),
                        "synthetic_quality_improved": bool(
                            severity[index] == 0.0 or gu_after[index] > gu_before[index]
                        ),
                        "actual_rtx_performed": False,
                    })
    return rows, representative, state_maps, config


def summarize(rows, seeds, target_gu, clearcoat_safety):
    seed_summaries = []
    for seed in seeds:
        selected = [row for row in rows if row["seed"] == seed]
        defective = [row for row in selected if row["defect_severity"] > 0.0]
        reasons = []
        if not all(row["gu_proxy_target_pass"] for row in selected):
            reasons.append("GU_PROXY_TARGET")
        if not all(row["clearcoat_safety_pass"] for row in selected):
            reasons.append("CLEARCOAT_SAFETY")
        if not all(row["synthetic_quality_improved"] for row in selected):
            reasons.append("NO_SYNTHETIC_IMPROVEMENT")
        seed_summaries.append({
            "seed": seed,
            "cell_count": len(selected),
            "defective_cell_count": len(defective),
            "gu_proxy_mean_before": float(np.mean([row["gu_proxy_before"] for row in selected])),
            "gu_proxy_mean_after": float(np.mean([row["gu_proxy_after"] for row in selected])),
            "gu_proxy_pass_count": sum(row["gu_proxy_target_pass"] for row in selected),
            "clearcoat_safety_pass_count": sum(row["clearcoat_safety_pass"] for row in selected),
            "all_targets_pass_count": sum(row["all_targets_pass"] for row in selected),
            "improved_defective_cell_count": sum(
                row["gu_proxy_after"] > row["gu_proxy_before"] for row in defective
            ),
            "minimum_clearcoat_after_um": min(row["clearcoat_after_um"] for row in selected),
            "maximum_clearcoat_removed_um": max(row["clearcoat_removed_um"] for row in selected),
            "failure_reasons": reasons,
            "passed": not reasons,
        })
    return {
        "status": "vehicle_multi_seed_synthetic_repeatability_prepared",
        "data_origin": "synthetic_vehicle_repeatability_not_rl",
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "region_count_per_seed": len(REGION_ORDER),
        "cell_count_per_seed": 150,
        "total_cell_count": len(rows),
        "target_gu_proxy": target_gu,
        "actual_gloss_meter_calibrated": False,
        "clearcoat_safety_limit_um": clearcoat_safety,
        "clearcoat_safety_evidence_tag": "PT-DESIGN",
        "actual_rtx_status": "representative_plan_prepared_not_run",
        "seed_summaries": seed_summaries,
        "passed_seed_count": sum(item["passed"] for item in seed_summaries),
        "passed": all(item["passed"] for item in seed_summaries),
        "important": (
            "All polishing/quality states are seeded synthetic design data, not RL "
            "or physical measurements. GU proxy and actual RTX must remain separate."
        ),
    }


def save_plot(path, seed_summaries, target_gu, clearcoat_safety):
    seeds = [str(item["seed"])[-4:] for item in seed_summaries]
    x = np.arange(len(seeds))
    before = [item["gu_proxy_mean_before"] for item in seed_summaries]
    after = [item["gu_proxy_mean_after"] for item in seed_summaries]
    pass_rate = [100.0 * item["all_targets_pass_count"] / item["cell_count"] for item in seed_summaries]
    minimum = [item["minimum_clearcoat_after_um"] for item in seed_summaries]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    axes[0].plot(x, before, "o-", label="before")
    axes[0].plot(x, after, "o-", label="after")
    axes[0].axhline(target_gu, color="crimson", linestyle="--", label="target")
    axes[0].set_title("Mean 20° GU proxy (not measured GU)")
    axes[0].legend()
    axes[1].bar(x, pass_rate, color="#22c55e")
    axes[1].set_ylim(0.0, 105.0)
    axes[1].set_title("All-target pass rate (%)")
    axes[2].plot(x, minimum, "o-", color="#7c3aed")
    axes[2].axhline(clearcoat_safety, color="crimson", linestyle="--")
    axes[2].set_title("Minimum clearcoat after (μm)")
    for axis in axes:
        axis.set_xticks(x, seeds)
        axis.set_xlabel("seed suffix")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("BMW 6-region, 150-cell multi-seed repeatability (synthetic, not RL)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(
    geometry_csv,
    output_dir,
    seeds,
    clearcoat_min=40.0,
    clearcoat_max=50.0,
    clearcoat_safety=35.0,
    target_gu=70.0,
):
    geometry = read_geometry(geometry_csv)
    rows, representatives, state_maps, config = generate_rows(
        geometry, seeds, clearcoat_min, clearcoat_max, clearcoat_safety, target_gu
    )
    output_dir = Path(output_dir)
    state_dir = output_dir / "states"
    state_dir.mkdir(parents=True, exist_ok=True)

    cells_path = output_dir / "vehicle_seed_repeatability_cells.csv"
    with cells_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    representative_path = output_dir / "representative_rtx_plan.csv"
    with representative_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(representatives[0]))
        writer.writeheader()
        writer.writerows(representatives)

    for (seed, region_id), maps in state_maps.items():
        directory = state_dir / f"seed_{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / f"{region_id}_state_maps.npz", **maps)

    summary = summarize(rows, seeds, target_gu, clearcoat_safety)
    summary["gu_proxy_model"] = config.metadata()
    summary["representative_rtx_planned_cell_count"] = len(representatives)
    summary_path = output_dir / "vehicle_seed_repeatability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    seed_summary_path = output_dir / "seed_summary.csv"
    flat_seed_summaries = [
        {**item, "failure_reasons": ";".join(item["failure_reasons"])}
        for item in summary["seed_summaries"]
    ]
    with seed_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_seed_summaries[0]))
        writer.writeheader()
        writer.writerows(flat_seed_summaries)

    plot_path = output_dir / "vehicle_seed_repeatability_summary.png"
    save_plot(plot_path, summary["seed_summaries"], target_gu, clearcoat_safety)
    return summary, cells_path, seed_summary_path, representative_path, summary_path, plot_path


def main():
    args = parse_args()
    outputs = run(
        args.geometry_csv,
        args.output_dir,
        args.seeds_parsed,
        args.clearcoat_min_um,
        args.clearcoat_max_um,
        args.clearcoat_safety_limit_um,
        args.target_gu_proxy,
    )
    summary, cells_path, seed_summary_path, plan_path, summary_path, plot_path = outputs
    print("")
    print("=" * 92)
    print("BMW 6영역 × 150셀 다중 랜덤 시드 반복성 검사 — 합성 상태, RL 아님")
    for item in summary["seed_summaries"]:
        reasons = ",".join(item["failure_reasons"]) or "없음"
        print(
            f"seed={item['seed']} | GU proxy {item['gu_proxy_mean_before']:.2f} → "
            f"{item['gu_proxy_mean_after']:.2f} | 목표={item['gu_proxy_pass_count']}/150 | "
            f"Clearcoat={item['clearcoat_safety_pass_count']}/150 | "
            f"min={item['minimum_clearcoat_after_um']:.3f} μm | "
            f"판정={'통과' if item['passed'] else '실패'} | 실패원인={reasons}"
        )
    print(
        f"최종: {summary['passed_seed_count']}/{summary['seed_count']}개 시드 통과, "
        f"총 {summary['total_cell_count']}셀"
    )
    print("주의: GU는 문헌 proxy이며 실제 Gloss Meter GU가 아님. 실제 RTX는 아직 실행 전.")
    print(f"셀 CSV       : {cells_path}")
    print(f"시드 CSV     : {seed_summary_path}")
    print(f"대표 RTX 계획: {plan_path}")
    print(f"요약 JSON    : {summary_path}")
    print(f"그래프       : {plot_path}")
    print("=" * 92)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
