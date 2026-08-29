#!/usr/bin/env python3
"""Aggregate geometry and actual RTX evidence for BMW body regions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REGION_IDS = (
    "hood",
    "roof",
    "negative_x_door",
    "positive_x_door",
    "negative_x_front_fender",
    "positive_x_front_fender",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_plot(path, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [item["region_id"] for item in summaries]
    means = np.asarray([item["hdr_mean"] for item in summaries])
    minima = np.asarray([item["hdr_min"] for item in summaries])
    maxima = np.asarray([item["hdr_max"] for item in summaries])
    x = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    axis.bar(x, means, color="#3b82f6", alpha=0.85, label="HDR ROI mean")
    axis.errorbar(
        x, means, yerr=np.vstack((means - minima, maxima - means)),
        fmt="none", ecolor="#111827", capsize=5, label="cell min/max",
    )
    axis.set_xticks(x, names, rotation=20, ha="right")
    axis.set_ylabel("HdrColor ROI intensity (not GU)")
    axis.set_title("BMW Z4 multi-region actual RTX Local 20° measurement")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def aggregate(results_root, output_dir):
    geometry_path = (
        results_root / "vehicle_multi_region_local_20"
        / "vehicle_multi_region_local_20_summary.json"
    )
    geometry = json.loads(geometry_path.read_text())
    summaries = []
    combined_rows = []
    for region_id in REGION_IDS:
        tag = f"vehicle_multi_region_{region_id}_rtx"
        directory = results_root / tag
        summary = json.loads((directory / "vehicle_mesh_rtx_summary.json").read_text())
        rows = read_csv(directory / "vehicle_mesh_rtx_results.csv")
        if summary.get("region_id") != region_id or len(rows) != 25:
            raise ValueError(f"{region_id}: mismatching or incomplete RTX result")
        values = np.asarray([float(row["hdr_roi_mean_intensity"]) for row in rows])
        region_passed = bool(
            summary["passed"]
            and np.all(np.isfinite(values))
            and np.all(values > 0.0)
        )
        summaries.append({
            "region_id": region_id,
            "region_label": summary["region_label"],
            "result_tag": tag,
            "sample_count": len(rows),
            "hdr_min": float(values.min()),
            "hdr_mean": float(values.mean()),
            "hdr_max": float(values.max()),
            "azimuth_retry_count": int(summary["azimuth_retry_count"]),
            "max_incident_angle_error_deg": float(
                summary["max_incident_angle_error_deg"]
            ),
            "max_detection_angle_error_deg": float(
                summary["max_detection_angle_error_deg"]
            ),
            "passed": region_passed,
        })
        for row in rows:
            combined_rows.append({"region_id": region_id, **row})

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_csv_path = output_dir / "vehicle_multi_region_rtx_cells.csv"
    with combined_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0]))
        writer.writeheader()
        writer.writerows(combined_rows)
    payload = {
        "status": "bmw_multi_region_actual_rtx_final_report",
        "asset": geometry["asset"],
        "target_prim": geometry["target_prim"],
        "geometry_region_passed_count": geometry["region_passed_count"],
        "geometry_region_count": geometry["region_count"],
        "actual_rtx_region_passed_count": sum(item["passed"] for item in summaries),
        "actual_rtx_region_count": len(summaries),
        "actual_rtx_sample_count": sum(item["sample_count"] for item in summaries),
        "renderer": "RTX Path Tracing",
        "measurement_aov": "HdrColor",
        "is_gu": False,
        "regions": summaries,
        "passed": bool(
            geometry["passed"] and all(item["passed"] for item in summaries)
        ),
        "important": (
            "This proves Local 20-degree geometry and positive finite RTX HdrColor "
            "sampling on the selected BMW regions. It is not GU calibration and "
            "does not yet apply polishing before/after states to every region."
        ),
    }
    json_path = output_dir / "vehicle_multi_region_final_report.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "BMW Z4 다중 영역 Local 20° 실제 RTX 최종 보고서",
        f"기하 검증: {payload['geometry_region_passed_count']}/{payload['geometry_region_count']} 통과",
        f"실제 RTX 영역: {payload['actual_rtx_region_passed_count']}/{payload['actual_rtx_region_count']} 통과",
        f"실제 RTX 측정점: {payload['actual_rtx_sample_count']}/150",
        "측정량: HdrColor ROI intensity (GU 아님)",
        "",
    ]
    for item in summaries:
        lines.append(
            f"{item['region_id']}: HDR {item['hdr_min']:.6f} ~ "
            f"{item['hdr_max']:.6f}, mean={item['hdr_mean']:.6f}, "
            f"azimuth_retry={item['azimuth_retry_count']}, "
            f"passed={item['passed']}"
        )
    lines.extend([
        "",
        "주의: 영역마다 현재 상태의 Clearcoat roughness=0.10을 사용한 단일 상태 검사다.",
        "각 영역의 폴리싱 전·후 결함 개선 비교나 실제 Gloss Meter GU 보정은 아니다.",
    ])
    text_path = output_dir / "vehicle_multi_region_final_report.txt"
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_path = output_dir / "vehicle_multi_region_hdr_summary.png"
    save_plot(plot_path, summaries)
    return payload, json_path, text_path, plot_path


def main():
    args = parse_args()
    payload, json_path, text_path, plot_path = aggregate(
        args.results_root, args.output_dir
    )
    print("")
    print("BMW Z4 다중 영역 실제 RTX 집계")
    for item in payload["regions"]:
        print(
            f"  {item['region_id']:<28} HDR mean={item['hdr_mean']:.6f}, "
            f"25/25, {'통과' if item['passed'] else '실패'}"
        )
    print(
        f"최종: {payload['actual_rtx_region_passed_count']}/"
        f"{payload['actual_rtx_region_count']} 영역 통과"
    )
    print(f"JSON: {json_path}")
    print(f"TXT : {text_path}")
    print(f"Plot: {plot_path}")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
