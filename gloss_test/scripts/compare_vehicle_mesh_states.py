#!/usr/bin/env python3
"""Compare actual-vehicle RTX scans before and after simulated polishing."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-dir", type=Path, required=True)
    parser.add_argument("--after-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    args = parse_args()
    before = read(args.before_dir / "vehicle_mesh_rtx_results.csv")
    after = read(args.after_dir / "vehicle_mesh_rtx_results.csv")
    if len(before) != 25 or len(after) != 25:
        raise RuntimeError("Expected matching 5x5 vehicle scans")
    combined = []
    for before_row, after_row in zip(before, after):
        before_cell = (int(before_row["grid_row"]), int(before_row["grid_column"]))
        after_cell = (int(after_row["grid_row"]), int(after_row["grid_column"]))
        if before_cell != after_cell:
            raise RuntimeError("Before/after vehicle cell ordering differs")
        before_position = np.asarray([
            float(before_row[f"position_{axis}_m"]) for axis in "xyz"
        ])
        after_position = np.asarray([
            float(after_row[f"position_{axis}_m"]) for axis in "xyz"
        ])
        if np.linalg.norm(before_position - after_position) > 1e-8:
            raise RuntimeError(f"Before/after positions differ at {before_cell}")
        relative_before = float(before_row["relative_gloss_to_center_not_gu"])
        relative_after = float(after_row["relative_gloss_to_center_not_gu"])
        combined.append({
            "grid_row": before_cell[0],
            "grid_column": before_cell[1],
            "initial_severity": float(before_row["initial_severity"]),
            "relative_gloss_before_not_gu": relative_before,
            "relative_gloss_after_not_gu": relative_after,
            "relative_gloss_improvement_not_gu": relative_after - relative_before,
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "before_after_cells.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)

    before_grid = np.asarray([
        row["relative_gloss_before_not_gu"] for row in combined
    ]).reshape(5, 5)
    after_grid = np.asarray([
        row["relative_gloss_after_not_gu"] for row in combined
    ]).reshape(5, 5)
    improvement = after_grid - before_grid
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), constrained_layout=True)
    for axis, data, title, colour_map in (
        (axes[0], before_grid, "Vehicle hood before\nrelative gloss, not GU", "viridis"),
        (axes[1], after_grid, "Vehicle hood after\nrelative gloss, not GU", "viridis"),
        (axes[2], improvement, "Vehicle hood improvement\nafter - before", "RdYlGn"),
    ):
        image = axis.imshow(data, origin="lower", cmap=colour_map)
        axis.set_title(title)
        axis.set_xlabel("grid column")
        axis.set_ylabel("grid row")
        axis.set_xticks(range(5), range(1, 6))
        axis.set_yticks(range(5), range(1, 6))
        for row in range(5):
            for column in range(5):
                axis.text(
                    column, row, f"{data[row, column]:.2f}",
                    ha="center", va="center", fontsize=8,
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    plot_path = args.output_dir / "vehicle_before_after_heatmaps.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    non_pristine = [row for row in combined if row["initial_severity"] > 0.0]
    pristine = [row for row in combined if row["initial_severity"] == 0.0]
    if len(pristine) != 1:
        raise RuntimeError(f"Expected exactly one pristine cell, got {len(pristine)}")
    before_mean = float(np.mean([
        row["relative_gloss_before_not_gu"] for row in non_pristine
    ]))
    after_mean = float(np.mean([
        row["relative_gloss_after_not_gu"] for row in non_pristine
    ]))
    improved_count = sum(
        row["relative_gloss_improvement_not_gu"] > 0.02 for row in non_pristine
    )
    pristine_change = pristine[0]["relative_gloss_improvement_not_gu"]
    summary = {
        "mode": "actual_vehicle_mesh_distributed_defect_before_after_rtx",
        "metric": "relative_gloss_to_pristine_center_not_gu",
        "non_pristine_cell_count": len(non_pristine),
        "mean_before_non_pristine": before_mean,
        "mean_after_non_pristine": after_mean,
        "mean_improvement_non_pristine": after_mean - before_mean,
        "cells_improved_by_more_than_0_02": improved_count,
        "pristine_cell": [pristine[0]["grid_row"], pristine[0]["grid_column"]],
        "pristine_relative_change": pristine_change,
    }
    summary["passed"] = (
        after_mean > before_mean + 0.05
        and improved_count >= 20
        and abs(pristine_change) <= 0.03
    )
    summary_path = args.output_dir / "vehicle_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("")
    print("실제 차량 보닛 분산결함 폴리싱 전·후 비교")
    print(f"  비정상부 평균 상대광택: {before_mean:.6f} -> {after_mean:.6f}")
    print(f"  0.02 초과 개선 셀     : {improved_count}/{len(non_pristine)}")
    print(f"  정상부 상대변화       : {pristine_change:+.6f}")
    print(f"  최종 판정             : {'통과' if summary['passed'] else '실패'}")
    print(f"  비교 CSV              : {csv_path}")
    print(f"  히트맵                : {plot_path}")
    if not summary["passed"]:
        raise RuntimeError(f"Vehicle comparison failed; see {summary_path}")


if __name__ == "__main__":
    main()
