"""Compare Gate F5 vertical and local-normal pad control using frozen checks."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from typing import Any

import numpy as np

from learning.rl.gate_f_curved_physx_summarize import evaluate_summary


EXPECTED_KINDS = ("flat", "cylinder", "sphere", "freeform")
F5_THRESHOLDS = {
    "target_quaternion_norm_max_error": 1.0e-5,
    "target_quaternion_step_angle_max_deg": 1.0,
    "target_tilt_max_deg": 15.0,
    "normal_alignment_error_steady_p95_deg": 2.0,
    "curved_alignment_error_ratio_max": 0.25,
    "curved_removal_relative_delta_max": 0.05,
    "coverage_drop_max": 0.01,
    "flat_force_steady_mean_abs_delta_n": 0.10,
    "flat_gap_p95_abs_delta_m": 0.0002,
    "flat_removal_abs_delta_um": 0.005,
    "flat_coverage_abs_delta": 0.01,
    "flat_target_tilt_max_deg": 1.0e-6,
}


def _read_summary(directory: str) -> dict[str, Any]:
    with open(os.path.join(directory, "summary.json"), encoding="utf-8") as handle:
        return json.load(handle)


def _vertical_tilt_p95(directory: str, steady_start: int) -> float:
    values = []
    with open(os.path.join(directory, "force_trace.csv"), newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index >= steady_start:
                nz = float(row["surface_normal_z"])
                values.append(np.degrees(np.arccos(np.clip(nz, -1.0, 1.0))))
    if not values:
        raise ValueError(f"no steady trace values in {directory}")
    return float(np.percentile(values, 95))


def _check(kind: str, name: str, observed: Any, operator: str, threshold: Any) -> dict:
    if observed is None:
        passed = False
    elif operator == "le":
        passed = observed <= threshold
    elif operator == "ge":
        passed = observed >= threshold
    elif operator == "gt":
        passed = observed > threshold
    elif operator == "eq":
        passed = observed == threshold
    else:
        raise ValueError(operator)
    return {
        "surface_kind": kind,
        "check": name,
        "operator": operator,
        "threshold": threshold,
        "observed": observed,
        "passed": passed,
    }


def evaluate_pair(
    vertical: dict[str, Any], normal: dict[str, Any], vertical_tilt_p95_deg: float
) -> list[dict[str, Any]]:
    kind = normal["surface_kind"]
    rows = []
    for row in evaluate_summary(normal):
        rows.append(_check(
            kind, "normal_F4_" + row["metric"], row["observed"],
            row["operator"], row["threshold"],
        ))
    for metric in (
        "target_quaternion_norm_max_error",
        "target_quaternion_step_angle_max_deg",
        "target_tilt_max_deg",
        "normal_alignment_error_steady_p95_deg",
    ):
        rows.append(_check(kind, metric, normal.get(metric), "le", F5_THRESHOLDS[metric]))

    removal_delta = abs(normal["roi_removal_mean_um"] - vertical["roi_removal_mean_um"])
    removal_relative = removal_delta / max(abs(vertical["roi_removal_mean_um"]), 1.0e-12)
    coverage_delta = normal["roi_coverage_fraction"] - vertical["roi_coverage_fraction"]
    if kind == "flat":
        rows.extend((
            _check(kind, "flat_target_tilt_max_deg", normal["target_tilt_max_deg"], "le",
                   F5_THRESHOLDS["flat_target_tilt_max_deg"]),
            _check(kind, "flat_force_steady_mean_abs_delta_n", abs(
                normal["force_sensor_filtered_steady_mean_n"]
                - vertical["force_sensor_filtered_steady_mean_n"]), "le",
                F5_THRESHOLDS["flat_force_steady_mean_abs_delta_n"]),
            _check(kind, "flat_gap_p95_abs_delta_m", abs(
                normal["gap_tracking_abs_error_steady_p95_m"]
                - vertical["gap_tracking_abs_error_steady_p95_m"]), "le",
                F5_THRESHOLDS["flat_gap_p95_abs_delta_m"]),
            _check(kind, "flat_removal_abs_delta_um", removal_delta, "le",
                   F5_THRESHOLDS["flat_removal_abs_delta_um"]),
            _check(kind, "flat_coverage_abs_delta", abs(coverage_delta), "le",
                   F5_THRESHOLDS["flat_coverage_abs_delta"]),
        ))
    else:
        ratio = normal["normal_alignment_error_steady_p95_deg"] / max(
            vertical_tilt_p95_deg, 1.0e-12)
        rows.extend((
            _check(kind, "curved_alignment_error_ratio", ratio, "le",
                   F5_THRESHOLDS["curved_alignment_error_ratio_max"]),
            _check(kind, "curved_removal_relative_delta", removal_relative, "le",
                   F5_THRESHOLDS["curved_removal_relative_delta_max"]),
            _check(kind, "curved_coverage_delta", coverage_delta, "ge",
                   -F5_THRESHOLDS["coverage_drop_max"]),
        ))
    return rows


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(vertical_dirs: list[str], normal_dirs: list[str], out_dir: str) -> dict:
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite output directory: {out_dir}")
    vertical = {_read_summary(path)["surface_kind"]: (path, _read_summary(path))
                for path in vertical_dirs}
    normal = {_read_summary(path)["surface_kind"]: (path, _read_summary(path))
              for path in normal_dirs}
    if set(vertical) != set(EXPECTED_KINDS) or set(normal) != set(EXPECTED_KINDS):
        raise ValueError("vertical and normal inputs must each contain all four surface kinds")
    checks, pairs = [], []
    for kind in EXPECTED_KINDS:
        vertical_path, v = vertical[kind]
        normal_path, n = normal[kind]
        if v.get("normal_aligned_pad") or not n.get("normal_aligned_pad"):
            raise ValueError(f"alignment mode mismatch for {kind}")
        tilt_p95 = _vertical_tilt_p95(vertical_path, int(v["steady_start_control_step"]))
        rows = evaluate_pair(v, n, tilt_p95)
        checks.extend(rows)
        pairs.append({
            "surface_kind": kind,
            "vertical_source": os.path.abspath(vertical_path),
            "normal_source": os.path.abspath(normal_path),
            "vertical_surface_tilt_p95_deg": tilt_p95,
            "normal_alignment_error_p95_deg": n["normal_alignment_error_steady_p95_deg"],
            "vertical_force_error_p95_n": v["force_tracking_abs_error_steady_p95_n"],
            "normal_force_error_p95_n": n["force_tracking_abs_error_steady_p95_n"],
            "vertical_raw_force_max_n": v["force_sensor_raw_max_n"],
            "normal_raw_force_max_n": n["force_sensor_raw_max_n"],
            "vertical_removal_mean_um": v["roi_removal_mean_um"],
            "normal_removal_mean_um": n["roi_removal_mean_um"],
            "vertical_coverage": v["roi_coverage_fraction"],
            "normal_coverage": n["roi_coverage_fraction"],
            "checks_passed": sum(bool(row["passed"]) for row in rows),
            "checks_total": len(rows),
            "pair_pass": all(bool(row["passed"]) for row in rows),
        })
    passed = all(row["pair_pass"] for row in pairs)
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F5",
        "decision": "PASS" if passed else "FAIL",
        "pairs_passed": sum(bool(row["pair_pass"]) for row in pairs),
        "pair_count": len(pairs),
        "all_pairs_pass": passed,
        "thresholds_frozen_before_physx": F5_THRESHOLDS,
        "training_performed": False,
        "next_step": "Further Gate F validation requires separate user approval.",
    }
    os.makedirs(out_dir)
    _write_csv(os.path.join(out_dir, "pair_summary.csv"), pairs)
    _write_csv(os.path.join(out_dir, "acceptance_checks.csv"), checks)
    with open(os.path.join(out_dir, "decision.json"), "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertical-dir", action="append", required=True)
    parser.add_argument("--normal-dir", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.vertical_dir, args.normal_dir, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
