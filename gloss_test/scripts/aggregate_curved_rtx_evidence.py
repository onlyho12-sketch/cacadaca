#!/usr/bin/env python3
"""Aggregate existing real RTX before/after evidence by surface family."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_PAIRS = (
    ("plane", "generalization_plane_initial", "generalization_plane_improved"),
    ("convex_cylinder", "cylinder_white_clearcoat_initial", "cylinder_white_clearcoat_improved"),
    (
        "concave_cylinder",
        "generalization_concave_cylinder_initial",
        "generalization_concave_cylinder_improved",
    ),
    ("convex_sphere", "sphere_white_clearcoat_initial", "sphere_white_clearcoat_improved"),
    ("freeform_mild", "curved_distributed_initial", "curved_distributed_improved"),
    (
        "freeform_strong",
        "generalization_freeform_strong_initial",
        "generalization_freeform_strong_improved",
    ),
)


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate(results_root, output_dir):
    results_root = Path(results_root)
    summaries = []
    for scenario_id, before_tag, after_tag in DEFAULT_PAIRS:
        before_path = results_root / before_tag / "curved_rtx_gloss_results.csv"
        after_path = results_root / after_tag / "curved_rtx_gloss_results.csv"
        before = read(before_path)
        after = read(after_path)
        if len(before) != 25 or len(after) != 25:
            raise ValueError(f"{scenario_id}: expected matching 5x5 RTX results")
        before_hdr = np.asarray([float(row["hdr_roi_mean_intensity"]) for row in before])
        after_hdr = np.asarray([float(row["hdr_roi_mean_intensity"]) for row in after])
        if not np.all(np.isfinite(before_hdr)) or not np.all(np.isfinite(after_hdr)):
            raise ValueError(f"{scenario_id}: non-finite RTX values")
        improved = after_hdr > before_hdr
        before_summary = json.loads(
            (results_root / before_tag / "curved_rtx_summary.json").read_text()
        )
        after_summary = json.loads(
            (results_root / after_tag / "curved_rtx_summary.json").read_text()
        )
        summaries.append({
            "scenario_id": scenario_id,
            "rtx_before_tag": before_tag,
            "rtx_after_tag": after_tag,
            "sample_count": 25,
            "hdr_mean_before": float(before_hdr.mean()),
            "hdr_mean_after": float(after_hdr.mean()),
            "hdr_mean_improvement_percent": float(
                (after_hdr.mean() / before_hdr.mean() - 1.0) * 100.0
            ),
            "hdr_improved_cell_count": int(improved.sum()),
            "local_20_before_passed": bool(before_summary["passed"]),
            "local_20_after_passed": bool(after_summary["passed"]),
            "rtx_measurement_performed": True,
        })
    payload = {
        "status": "existing_curved_rtx_evidence_aggregated",
        "renderer": "RTX Path Tracing",
        "metric": "HdrColor ROI, not GU",
        "scenario_count_with_rtx": len(summaries),
        "scenario_count_without_rtx": 0,
        "scenarios": summaries,
        "passed": all(
            item["local_20_before_passed"]
            and item["local_20_after_passed"]
            and item["hdr_improved_cell_count"] >= 20
            for item in summaries
        ),
        "important": (
            "RTX before/after evidence exists for all six benchmark surface families."
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "surface_generalization_rtx_evidence.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    geometry_path = output_dir / "surface_generalization_summary.json"
    if geometry_path.is_file():
        geometry = json.loads(geometry_path.read_text())
        rtx_by_id = {item["scenario_id"]: item for item in summaries}
        coverage = []
        for item in geometry["scenarios"]:
            scenario_id = item["scenario_id"]
            rtx = rtx_by_id.get(scenario_id)
            coverage.append({
                "scenario_id": scenario_id,
                "geometry_local_20_passed": (
                    item["max_incident_angle_error_deg"] < 1.0e-6
                    and item["max_detection_angle_error_deg"] < 1.0e-6
                    and item["max_reflection_error_deg"] < 1.0e-6
                ),
                "synthetic_quality_state_prepared": True,
                "actual_rtx_before_after_performed": rtx is not None,
                "actual_rtx_passed": bool(
                    rtx is not None
                    and rtx["local_20_before_passed"]
                    and rtx["local_20_after_passed"]
                    and rtx["hdr_improved_cell_count"] >= 20
                ),
                "validation_level": (
                    "geometry_synthetic_and_actual_rtx"
                    if rtx is not None else "geometry_and_synthetic_only"
                ),
            })
        final = {
            "status": "surface_generalization_final_report",
            "data_origin": "synthetic_surface_generalization_not_rl",
            "scenario_count": len(coverage),
            "geometry_passed_scenario_count": sum(
                item["geometry_local_20_passed"] for item in coverage
            ),
            "actual_rtx_performed_scenario_count": sum(
                item["actual_rtx_before_after_performed"] for item in coverage
            ),
            "actual_rtx_passed_scenario_count": sum(
                item["actual_rtx_passed"] for item in coverage
            ),
            "full_rtx_coverage": all(
                item["actual_rtx_before_after_performed"] for item in coverage
            ),
            "available_evidence_passed": bool(
                geometry["geometry_passed"] and payload["passed"]
            ),
            "coverage": coverage,
            "important": (
                "Synthetic quality values are not RL output. Actual RTX before/after "
                "evidence is available for all six surface families."
            ),
        }
        final_path = output_dir / "surface_generalization_final_report.json"
        final_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        report_lines = [
            "다중 표면 일반화 최종 검증 보고서",
            "데이터 출처: synthetic_surface_generalization_not_rl",
            f"기하 Local 20° 통과: {final['geometry_passed_scenario_count']}/{len(coverage)}",
            f"실제 RTX 전후 수행/통과: {final['actual_rtx_performed_scenario_count']}/"
            f"{len(coverage)} / {final['actual_rtx_passed_scenario_count']} 통과",
            f"전체 RTX 커버리지: {final['full_rtx_coverage']}",
            "",
        ]
        for item in coverage:
            report_lines.append(
                f"{item['scenario_id']}: {item['validation_level']}, "
                f"geometry={item['geometry_local_20_passed']}, "
                f"RTX={item['actual_rtx_passed']}"
            )
        report_lines.extend([
            "",
            "주의: 품질·Clearcoat 값은 합성 설계 데이터이며 실제 RL 출력이나 실측값이 아님.",
        ])
        (output_dir / "surface_generalization_final_report.txt").write_text(
            "\n".join(report_lines) + "\n", encoding="utf-8"
        )
    return path, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    path, payload = aggregate(args.results_root, args.output_dir)
    print("")
    print("기존 실제 RTX 곡면 증거 집계")
    for item in payload["scenarios"]:
        print(
            f"  {item['scenario_id']:<18} HDR "
            f"{item['hdr_mean_before']:.6f} -> {item['hdr_mean_after']:.6f}, "
            f"개선 {item['hdr_improved_cell_count']}/25"
        )
    print(f"RTX 증거 판정: {'통과' if payload['passed'] else '실패'}")
    print(f"JSON: {path}")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
