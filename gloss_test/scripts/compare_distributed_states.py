#!/usr/bin/env python3
"""Compare initial and polished whole-panel spatial gloss scans."""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: compare_distributed_states.py INITIAL_DIR IMPROVED_DIR OUTPUT_DIR")
    initial_dir, improved_dir, output_dir = map(Path, sys.argv[1:])
    output_dir.mkdir(parents=True, exist_ok=True)
    initial = read_rows(initial_dir / "spatial_gloss_results.csv")
    improved = read_rows(improved_dir / "spatial_gloss_results.csv")
    severity_rows = read_rows(initial_dir / "assets" / "severity_grid.csv")
    if not (len(initial) == len(improved) == len(severity_rows) == 25):
        raise RuntimeError("Expected exactly 25 matching cells in both scans and severity grid")

    combined = []
    for before, after, severity in zip(initial, improved, severity_rows):
        cell_before = (int(before["grid_row"]), int(before["grid_column"]))
        cell_after = (int(after["grid_row"]), int(after["grid_column"]))
        cell_severity = (int(severity["grid_row"]), int(severity["grid_column"]))
        if not cell_before == cell_after == cell_severity:
            raise RuntimeError("Grid cell ordering differs between inputs")
        relative_before = float(before["relative_to_center"])
        relative_after = float(after["relative_to_center"])
        combined.append({
            "grid_row": cell_before[0],
            "grid_column": cell_before[1],
            "initial_severity": float(severity["initial_severity"]),
            "relative_gloss_before_not_gu": relative_before,
            "relative_gloss_after_not_gu": relative_after,
            "relative_gloss_improvement_not_gu": relative_after - relative_before,
        })

    csv_path = output_dir / "before_after_cells.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0].keys()))
        writer.writeheader()
        writer.writerows(combined)

    before_grid = np.asarray(
        [row["relative_gloss_before_not_gu"] for row in combined], dtype=float
    ).reshape(5, 5)
    after_grid = np.asarray(
        [row["relative_gloss_after_not_gu"] for row in combined], dtype=float
    ).reshape(5, 5)
    improvement_grid = after_grid - before_grid
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), constrained_layout=True)
    panels = (
        (before_grid, "Before polishing\n(relative gloss, not GU)", "viridis", 0.0, 1.05),
        (after_grid, "After polishing\n(relative gloss, not GU)", "viridis", 0.0, 1.05),
        (improvement_grid, "Improvement\n(after - before)", "RdYlGn", None, None),
    )
    for axis, (data, title, colour_map, vmin, vmax) in zip(axes, panels):
        image = axis.imshow(data, origin="lower", cmap=colour_map, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("Grid column")
        axis.set_ylabel("Grid row")
        axis.set_xticks(range(5), range(1, 6))
        axis.set_yticks(range(5), range(1, 6))
        for row in range(5):
            for column in range(5):
                axis.text(column, row, f"{data[row, column]:.2f}", ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    plot_path = output_dir / "before_after_improvement_heatmaps.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    non_pristine = [row for row in combined if row["initial_severity"] > 0.0]
    improved_cells = sum(
        row["relative_gloss_improvement_not_gu"] > 0.02 for row in non_pristine
    )
    before_mean = float(np.mean([row["relative_gloss_before_not_gu"] for row in non_pristine]))
    after_mean = float(np.mean([row["relative_gloss_after_not_gu"] for row in non_pristine]))
    pristine = next(row for row in combined if row["initial_severity"] == 0.0)
    validation = {
        "metric": "relative_gloss_to_pristine_center",
        "is_gu": False,
        "grid": [5, 5],
        "pristine_cell": [pristine["grid_row"], pristine["grid_column"]],
        "non_pristine_cell_count": len(non_pristine),
        "cells_improved_by_more_than_0_02": improved_cells,
        "mean_before_non_pristine": before_mean,
        "mean_after_non_pristine": after_mean,
        "mean_improvement_non_pristine": after_mean - before_mean,
        "pristine_relative_before": pristine["relative_gloss_before_not_gu"],
        "pristine_relative_after": pristine["relative_gloss_after_not_gu"],
    }
    validation["passed"] = (
        validation["pristine_cell"] == [3, 3]
        and improved_cells >= 20
        and after_mean > before_mean + 0.05
        and 0.95 <= validation["pristine_relative_before"] <= 1.05
        and 0.95 <= validation["pristine_relative_after"] <= 1.05
    )
    summary_path = output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(f"[Distributed Comparison] {validation}")
    print(f"[Distributed Comparison] CSV: {csv_path}")
    print(f"[Distributed Comparison] plot: {plot_path}")
    if not validation["passed"]:
        raise RuntimeError(f"Distributed polishing comparison failed; see {summary_path}")


if __name__ == "__main__":
    main()
