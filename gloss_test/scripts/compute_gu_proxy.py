#!/usr/bin/env python3
"""Convert before/after relative gloss cells to a literature-derived GU proxy."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gu_proxy import (
    LiteratureGuProxyConfig,
    gu_proxy_passes,
    relative_gloss_to_gu_proxy,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="before_after_cells.csv produced by compare_distributed_states.py",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--defective-anchor-gu", type=float, default=25.0)
    parser.add_argument("--good-anchor-gu", type=float, default=78.0)
    parser.add_argument("--high-gloss-anchor-gu", type=float, default=89.0)
    parser.add_argument("--target-gu", type=float, default=70.0)
    return parser.parse_args()


def read_relative_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    required = {
        "grid_row",
        "grid_column",
        "initial_severity",
        "relative_gloss_before_not_gu",
        "relative_gloss_after_not_gu",
    }
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {sorted(missing)}")
    return rows


def convert_rows(rows, config):
    converted = []
    seen = set()
    for row in rows:
        grid_row = int(row["grid_row"])
        grid_column = int(row["grid_column"])
        cell = (grid_row, grid_column)
        if cell in seen:
            raise RuntimeError(f"Duplicate grid cell: {cell}")
        seen.add(cell)
        relative_before = float(row["relative_gloss_before_not_gu"])
        relative_after = float(row["relative_gloss_after_not_gu"])
        proxy_before = relative_gloss_to_gu_proxy(relative_before, config)
        proxy_after = relative_gloss_to_gu_proxy(relative_after, config)
        converted.append({
            "grid_row": grid_row,
            "grid_column": grid_column,
            "initial_severity": float(row["initial_severity"]),
            "relative_gloss_before_not_gu": relative_before,
            "relative_gloss_after_not_gu": relative_after,
            "predicted_20deg_gu_proxy_before": proxy_before,
            "predicted_20deg_gu_proxy_after": proxy_after,
            "predicted_20deg_gu_proxy_improvement": proxy_after - proxy_before,
            "before_target_70gu_pass": gu_proxy_passes(proxy_before, config),
            "after_target_70gu_pass": gu_proxy_passes(proxy_after, config),
        })
    return converted


def infer_grid_shape(rows):
    row_count = max(row["grid_row"] for row in rows)
    column_count = max(row["grid_column"] for row in rows)
    expected = row_count * column_count
    if len(rows) != expected:
        raise RuntimeError(
            f"Grid is incomplete: got {len(rows)} cells, expected {expected}"
        )
    return row_count, column_count


def grid_from_rows(rows, key, shape, dtype=float):
    grid = np.empty(shape, dtype=dtype)
    for row in rows:
        grid[row["grid_row"] - 1, row["grid_column"] - 1] = row[key]
    return grid


def statistics(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def create_summary(rows, config, shape):
    non_pristine = [row for row in rows if row["initial_severity"] > 0.0]
    pristine = [row for row in rows if row["initial_severity"] == 0.0]
    if len(pristine) != 1:
        raise RuntimeError(f"Expected one pristine cell, got {len(pristine)}")

    before_all = [row["predicted_20deg_gu_proxy_before"] for row in rows]
    after_all = [row["predicted_20deg_gu_proxy_after"] for row in rows]
    before_non = [row["predicted_20deg_gu_proxy_before"] for row in non_pristine]
    after_non = [row["predicted_20deg_gu_proxy_after"] for row in non_pristine]
    pass_before_all = sum(row["before_target_70gu_pass"] for row in rows)
    pass_after_all = sum(row["after_target_70gu_pass"] for row in rows)
    pass_before_non = sum(row["before_target_70gu_pass"] for row in non_pristine)
    pass_after_non = sum(row["after_target_70gu_pass"] for row in non_pristine)
    pristine_row = pristine[0]

    summary = {
        "model": config.metadata(),
        "input_metric": {
            "name": "relative_gloss_to_pristine_center",
            "is_gu": False,
        },
        "output_metric": {
            "name": config.output_metric,
            "is_measured_gu": False,
            "is_literature_proxy": True,
        },
        "grid": list(shape),
        "cell_count": len(rows),
        "pristine_cell": [pristine_row["grid_row"], pristine_row["grid_column"]],
        "non_pristine_cell_count": len(non_pristine),
        "before_all_cells": statistics(before_all),
        "after_all_cells": statistics(after_all),
        "before_non_pristine": statistics(before_non),
        "after_non_pristine": statistics(after_non),
        "mean_improvement_non_pristine": float(np.mean(after_non) - np.mean(before_non)),
        "target_pass_count_before_all": pass_before_all,
        "target_pass_count_after_all": pass_after_all,
        "target_pass_count_before_non_pristine": pass_before_non,
        "target_pass_count_after_non_pristine": pass_after_non,
        "target_pass_ratio_before_all": pass_before_all / len(rows),
        "target_pass_ratio_after_all": pass_after_all / len(rows),
        "target_pass_ratio_before_non_pristine": pass_before_non / len(non_pristine),
        "target_pass_ratio_after_non_pristine": pass_after_non / len(non_pristine),
        "pristine_proxy_before": pristine_row["predicted_20deg_gu_proxy_before"],
        "pristine_proxy_after": pristine_row["predicted_20deg_gu_proxy_after"],
    }
    summary["passed"] = (
        summary["before_non_pristine"]["mean"] < config.target_gu
        and summary["after_non_pristine"]["mean"] >= config.target_gu
        and summary["mean_improvement_non_pristine"] > 0.0
        and abs(summary["pristine_proxy_before"] - config.good_refinish_anchor_gu) < 1e-9
        and abs(summary["pristine_proxy_after"] - config.good_refinish_anchor_gu) < 1e-9
    )
    return summary


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def annotate_heatmap(axis, data, fmt, text_colors=None):
    for row in range(data.shape[0]):
        for column in range(data.shape[1]):
            color = "black" if text_colors is None else text_colors[row, column]
            axis.text(
                column,
                row,
                format(data[row, column], fmt),
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )


def save_plot(path, rows, shape, config):
    before = grid_from_rows(rows, "predicted_20deg_gu_proxy_before", shape)
    after = grid_from_rows(rows, "predicted_20deg_gu_proxy_after", shape)
    improvement = after - before
    passed = after >= config.target_gu

    figure, axes = plt.subplots(2, 2, figsize=(10.8, 9.0), constrained_layout=True)
    panels = (
        (axes[0, 0], before, "Before polishing\n20° GU literature proxy", "viridis", 20, 90),
        (axes[0, 1], after, "After polishing\n20° GU literature proxy", "viridis", 20, 90),
        (axes[1, 0], improvement, "GU proxy improvement\n(after - before)", "RdYlGn", 0, None),
    )
    for axis, data, title, colour_map, vmin, vmax in panels:
        image = axis.imshow(data, origin="lower", cmap=colour_map, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        annotate_heatmap(axis, data, ".1f")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    status_axis = axes[1, 1]
    status_image = status_axis.imshow(
        passed.astype(int),
        origin="lower",
        cmap=matplotlib.colors.ListedColormap(["#d73027", "#1a9850"]),
        vmin=0,
        vmax=1,
    )
    status_axis.set_title(f"After target map\nPASS if proxy GU ≥ {config.target_gu:.0f}")
    labels = np.where(passed, "PASS", "FAIL")
    for row in range(shape[0]):
        for column in range(shape[1]):
            status_axis.text(
                column,
                row,
                f"{labels[row, column]}\n{after[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                weight="bold",
            )
    figure.colorbar(status_image, ax=status_axis, ticks=[0, 1], fraction=0.046, pad=0.04)

    for axis in axes.flat:
        axis.set_xlabel("Grid column")
        axis.set_ylabel("Grid row")
        axis.set_xticks(range(shape[1]), range(1, shape[1] + 1))
        axis.set_yticks(range(shape[0]), range(1, shape[0] + 1))

    figure.suptitle(
        "Simulation GU proxy — literature anchored, not Gloss Meter calibrated",
        fontsize=13,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_npz(path, rows, shape, config):
    np.savez_compressed(
        path,
        gu_proxy_before=grid_from_rows(
            rows, "predicted_20deg_gu_proxy_before", shape
        ),
        gu_proxy_after=grid_from_rows(
            rows, "predicted_20deg_gu_proxy_after", shape
        ),
        gu_proxy_improvement=grid_from_rows(
            rows, "predicted_20deg_gu_proxy_improvement", shape
        ),
        target_pass_after=grid_from_rows(
            rows, "after_target_70gu_pass", shape, dtype=bool
        ),
        target_gu=np.asarray(config.target_gu),
    )


def save_text_report(path, summary):
    model = summary["model"]
    before = summary["before_non_pristine"]
    after = summary["after_non_pristine"]
    lines = [
        "PolyTwin 20-degree GU Literature Proxy Report",
        "=" * 48,
        f"Model: {model['model_id']}",
        f"Source DOI: {model['source_doi']}",
        "Actual Gloss Meter calibrated: NO",
        "Metric type: simulation-only literature proxy",
        "",
        f"Anchors: defective={model['defective_anchor_gu']:.1f}, "
        f"good_refinish={model['good_refinish_anchor_gu']:.1f}, "
        f"high_gloss={model['high_gloss_vehicle_anchor_gu']:.1f} GU",
        f"Target: {model['target_gu']:.1f} GU proxy",
        "",
        f"Non-pristine mean before: {before['mean']:.3f} GU proxy",
        f"Non-pristine mean after : {after['mean']:.3f} GU proxy",
        f"Mean improvement         : {summary['mean_improvement_non_pristine']:.3f}",
        f"Non-pristine pass cells before: "
        f"{summary['target_pass_count_before_non_pristine']}/"
        f"{summary['non_pristine_cell_count']}",
        f"Non-pristine pass cells after : "
        f"{summary['target_pass_count_after_non_pristine']}/"
        f"{summary['non_pristine_cell_count']}",
        f"Validation passed: {summary['passed']}",
        "",
        "WARNING: These values are not physical Gloss Meter measurements.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    config = LiteratureGuProxyConfig(
        defective_anchor_gu=args.defective_anchor_gu,
        good_refinish_anchor_gu=args.good_anchor_gu,
        high_gloss_vehicle_anchor_gu=args.high_gloss_anchor_gu,
        target_gu=args.target_gu,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = convert_rows(read_relative_rows(args.input_csv), config)
    shape = infer_grid_shape(rows)
    summary = create_summary(rows, config, shape)

    csv_path = args.output_dir / "gu_proxy_cells.csv"
    json_path = args.output_dir / "gu_proxy_summary.json"
    plot_path = args.output_dir / "gu_proxy_before_after_heatmaps.png"
    npz_path = args.output_dir / "gu_proxy_maps.npz"
    text_path = args.output_dir / "gu_proxy_report.txt"

    save_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_plot(plot_path, rows, shape, config)
    save_npz(npz_path, rows, shape, config)
    save_text_report(text_path, summary)

    print(
        "[GU Proxy] non-pristine mean "
        f"{summary['before_non_pristine']['mean']:.3f} -> "
        f"{summary['after_non_pristine']['mean']:.3f}; "
        f"target-pass cells "
        f"{summary['target_pass_count_after_non_pristine']}/"
        f"{summary['non_pristine_cell_count']}"
    )
    print(f"[GU Proxy] CSV: {csv_path}")
    print(f"[GU Proxy] JSON: {json_path}")
    print(f"[GU Proxy] plot: {plot_path}")
    if not summary["passed"]:
        raise RuntimeError(f"GU proxy validation failed; see {json_path}")


if __name__ == "__main__":
    main()
