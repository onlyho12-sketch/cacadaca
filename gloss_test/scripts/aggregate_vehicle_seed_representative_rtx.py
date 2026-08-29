#!/usr/bin/env python3
"""Select or aggregate representative actual-RTX cells for one vehicle seed."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REGION_ORDER = (
    "hood",
    "roof",
    "negative_x_door",
    "positive_x_door",
    "negative_x_front_fender",
    "positive_x_front_fender",
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def planned_rows(path, seed, region_id):
    rows = [
        row for row in read_csv(path)
        if int(row["seed"]) == int(seed) and row["region_id"] == region_id
    ]
    if not rows:
        raise ValueError(f"no representative plan for seed={seed}, region={region_id}")
    return rows


def planned_cells(path, seed, region_id):
    return [
        (int(row["grid_row"]), int(row["grid_column"]))
        for row in planned_rows(path, seed, region_id)
    ]


def _keyed(rows, label):
    keyed = {}
    for row in rows:
        key = (int(row["grid_row"]), int(row["grid_column"]))
        if key in keyed:
            raise ValueError(f"{label}: duplicate cell {key}")
        keyed[key] = row
    return keyed


def save_plot(path, rows):
    labels = [
        f"{row['region_id']}\n({row['grid_row']},{row['grid_column']})"
        for row in rows
    ]
    x = np.arange(len(rows))
    before = np.asarray([row["rtx_hdr_before"] for row in rows])
    after = np.asarray([row["rtx_hdr_after"] for row in rows])
    width = 0.38
    figure, axis = plt.subplots(figsize=(18.0, 6.5), constrained_layout=True)
    axis.bar(x - width / 2.0, before, width, label="before", color="#ef4444")
    axis.bar(x + width / 2.0, after, width, label="after", color="#22c55e")
    axis.set_xticks(x, labels, rotation=65, ha="right", fontsize=7)
    axis.set_ylabel("HdrColor ROI intensity (not GU)")
    axis.set_title("BMW representative actual RTX before/after by region")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def aggregate(results_root, state_root, plan_path, output_dir, seed):
    combined = []
    region_summaries = []
    for region_id in REGION_ORDER:
        plan = planned_rows(plan_path, seed, region_id)
        expected = {
            (int(row["grid_row"]), int(row["grid_column"])): row for row in plan
        }
        prefix = f"vehicle_seed_{seed}_{region_id}"
        before = _keyed(
            read_csv(results_root / f"{prefix}_before_representative" / "vehicle_mesh_rtx_results.csv"),
            f"{region_id} before",
        )
        after = _keyed(
            read_csv(results_root / f"{prefix}_after_representative" / "vehicle_mesh_rtx_results.csv"),
            f"{region_id} after",
        )
        if set(before) != set(expected) or set(after) != set(expected):
            raise ValueError(
                f"{region_id}: actual cells differ from plan; expected={sorted(expected)}, "
                f"before={sorted(before)}, after={sorted(after)}"
            )
        state_path = state_root / f"seed_{seed}" / f"{region_id}_state_maps.npz"
        with np.load(state_path) as archive:
            state = {name: np.asarray(archive[name]) for name in archive.files}
        region_rows = []
        for cell in expected:
            index = (cell[0] - 1, cell[1] - 1)
            before_row = before[cell]
            after_row = after[cell]
            expected_values = {
                "roughness_before": float(state["roughness_before"][index]),
                "roughness_after": float(state["roughness_after"][index]),
                "scratch_before": float(state["scratch_before"][index]),
                "scratch_after": float(state["scratch_after"][index]),
                "clearcoat_before_um": float(state["clearcoat_before_um"][index]),
                "clearcoat_after_um": float(state["clearcoat_after_um"][index]),
            }
            actual_values = {
                "roughness_before": float(before_row["state_roughness"]),
                "roughness_after": float(after_row["state_roughness"]),
                "scratch_before": float(before_row["state_scratch"]),
                "scratch_after": float(after_row["state_scratch"]),
                "clearcoat_before_um": float(before_row["state_clearcoat_um"]),
                "clearcoat_after_um": float(after_row["state_clearcoat_um"]),
            }
            state_match = all(
                np.isclose(actual_values[key], value, rtol=0.0, atol=1.0e-5)
                for key, value in expected_values.items()
            )
            hdr_before = float(before_row["hdr_roi_mean_intensity"])
            hdr_after = float(after_row["hdr_roi_mean_intensity"])
            severity = float(state["severity"][index])
            row = {
                "seed": seed,
                "region_id": region_id,
                "grid_row": cell[0],
                "grid_column": cell[1],
                "selection_reason": expected[cell]["selection_reason"],
                "defect_severity": severity,
                **expected_values,
                "gu_proxy_before": float(state["gu_proxy_before"][index]),
                "gu_proxy_after": float(state["gu_proxy_after"][index]),
                "rtx_hdr_before": hdr_before,
                "rtx_hdr_after": hdr_after,
                "rtx_hdr_ratio_after_over_before": hdr_after / hdr_before,
                "rtx_improved": bool(hdr_after > hdr_before),
                "rtx_improvement_required": bool(severity > 0.0),
                "state_to_material_match": bool(state_match),
                "positive_finite_hdr": bool(
                    np.isfinite(hdr_before) and np.isfinite(hdr_after)
                    and hdr_before > 0.0 and hdr_after > 0.0
                ),
                "is_gu": False,
            }
            combined.append(row)
            region_rows.append(row)
        required = [row for row in region_rows if row["rtx_improvement_required"]]
        region_summaries.append({
            "region_id": region_id,
            "representative_cell_count": len(region_rows),
            "defective_representative_count": len(required),
            "defective_rtx_improved_count": sum(row["rtx_improved"] for row in required),
            "positive_finite_count": sum(row["positive_finite_hdr"] for row in region_rows),
            "state_match_count": sum(row["state_to_material_match"] for row in region_rows),
            "rtx_hdr_mean_before": float(np.mean([row["rtx_hdr_before"] for row in region_rows])),
            "rtx_hdr_mean_after": float(np.mean([row["rtx_hdr_after"] for row in region_rows])),
            "passed": bool(
                all(row["positive_finite_hdr"] for row in region_rows)
                and all(row["state_to_material_match"] for row in region_rows)
                and all(row["rtx_improved"] for row in required)
            ),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    cells_path = output_dir / "representative_rtx_before_after_cells.csv"
    with cells_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    summary = {
        "status": "vehicle_seed_representative_actual_rtx_complete",
        "seed": seed,
        "renderer": "RTX Path Tracing",
        "rtx_metric": "HdrColor ROI intensity",
        "rtx_is_gu": False,
        "representative_strategy": "maximum defect, median defect, pristine, center reference",
        "region_count": len(REGION_ORDER),
        "representative_cell_count": len(combined),
        "regions": region_summaries,
        "passed_region_count": sum(row["passed"] for row in region_summaries),
        "passed": all(row["passed"] for row in region_summaries),
        "important": (
            "Actual RTX validates only the planned representative cells for this seed. "
            "It is separate from the 750-cell synthetic GU-proxy result."
        ),
    }
    summary_path = output_dir / "representative_rtx_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    repeatability_summary_path = state_root.parent / "vehicle_seed_repeatability_summary.json"
    if repeatability_summary_path.is_file():
        repeatability_summary = json.loads(repeatability_summary_path.read_text())
        repeatability_summary.update({
            "actual_rtx_status": "one_seed_representative_actual_rtx_complete",
            "actual_rtx_seed": seed,
            "actual_rtx_representative_cell_count": len(combined),
            "actual_rtx_region_passed_count": summary["passed_region_count"],
            "actual_rtx_region_count": summary["region_count"],
            "actual_rtx_passed": summary["passed"],
            "actual_rtx_summary": str(summary_path.resolve()),
        })
        repeatability_summary_path.write_text(
            json.dumps(repeatability_summary, indent=2), encoding="utf-8"
        )
    plot_path = output_dir / "representative_rtx_before_after.png"
    save_plot(plot_path, combined)
    report_path = output_dir / "representative_rtx_report.txt"
    lines = [
        f"BMW 다중 영역 대표 실제 RTX 전·후 검사 — seed {seed}",
        "측정량: HdrColor ROI intensity (GU 아님)",
        "",
    ]
    for row in region_summaries:
        lines.append(
            f"{row['region_id']}: defect improved "
            f"{row['defective_rtx_improved_count']}/{row['defective_representative_count']}, "
            f"positive={row['positive_finite_count']}/{row['representative_cell_count']}, "
            f"state-match={row['state_match_count']}/{row['representative_cell_count']}, "
            f"passed={row['passed']}"
        )
    lines.extend([
        "",
        f"최종: {summary['passed_region_count']}/{summary['region_count']} 영역 통과",
        "주의: 대표 셀 검사이며 실제 Gloss Meter GU가 아니다.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, cells_path, summary_path, plot_path, report_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cells = subparsers.add_parser("cells", help="print scanner cell selection")
    cells.add_argument("--plan", type=Path, required=True)
    cells.add_argument("--seed", type=int, required=True)
    cells.add_argument("--region", choices=REGION_ORDER, required=True)
    report = subparsers.add_parser("aggregate", help="aggregate actual RTX outputs")
    report.add_argument("--results-root", type=Path, required=True)
    report.add_argument("--state-root", type=Path, required=True)
    report.add_argument("--plan", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)
    report.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "cells":
        print(";".join(f"{row},{column}" for row, column in planned_cells(
            args.plan, args.seed, args.region
        )))
        return
    summary, cells, summary_path, plot, report = aggregate(
        args.results_root, args.state_root, args.plan, args.output_dir, args.seed
    )
    print("")
    print("=" * 88)
    print(f"BMW 대표 실제 RTX 반복성 검사 — seed {args.seed}")
    for row in summary["regions"]:
        print(
            f"{row['region_id']:<28} 결함 개선="
            f"{row['defective_rtx_improved_count']}/{row['defective_representative_count']}, "
            f"HDR 양수={row['positive_finite_count']}/{row['representative_cell_count']}, "
            f"상태일치={row['state_match_count']}/{row['representative_cell_count']}, "
            f"판정={'통과' if row['passed'] else '실패'}"
        )
    print(f"최종: {summary['passed_region_count']}/{summary['region_count']} 영역 통과")
    print("주의: 실제 RTX HdrColor이며 실제 GU가 아님.")
    print(f"셀 CSV : {cells}")
    print(f"JSON    : {summary_path}")
    print(f"그래프  : {plot}")
    print(f"보고서  : {report}")
    print("=" * 88)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
