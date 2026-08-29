#!/usr/bin/env python3
"""Validate Isaac Lab polishing output and append the literature GU proxy."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from gu_proxy import LiteratureGuProxyConfig, gu_proxy_passes, relative_gloss_to_gu_proxy


GEOMETRY_COLUMNS = (
    "position_x_m", "position_y_m", "position_z_m",
    "normal_x", "normal_y", "normal_z",
)
PROCESS_COLUMNS = ("force_n", "rpm", "feed_mm_s", "step_over_ratio", "pass_count")
SURFACE_COLUMNS = (
    "roughness_before", "roughness_after", "scratch_before", "scratch_after",
    "ra_before_um", "ra_after_um", "rz_before_um", "rz_after_um",
    "clearcoat_before_um", "clearcoat_after_um", "clearcoat_removed_um",
)
OPTICAL_COLUMNS = ("relative_gloss_before_not_gu", "relative_gloss_after_not_gu")
REQUIRED_COLUMNS = (
    "data_origin", "episode_id", "cell_id", "grid_row", "grid_column",
    *GEOMETRY_COLUMNS, *PROCESS_COLUMNS, *SURFACE_COLUMNS, *OPTICAL_COLUMNS,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-gu", type=float, default=70.0)
    parser.add_argument("--normal-tolerance", type=float, default=1e-3)
    parser.add_argument("--clearcoat-tolerance-um", type=float, default=1e-3)
    return parser.parse_args()


def _number(row, column, row_number):
    try:
        value = float(row[column])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: {column} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"row {row_number}: {column} must be finite")
    return value


def _integer(row, column, row_number, minimum):
    value = _number(row, column, row_number)
    if not value.is_integer() or value < minimum:
        raise ValueError(f"row {row_number}: {column} must be an integer >= {minimum}")
    return int(value)


def read_and_validate(path, normal_tolerance=1e-3, clearcoat_tolerance_um=1e-3):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: header is missing")
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing required columns: {missing}")
        source_rows = list(reader)
    if not source_rows:
        raise ValueError(f"{path}: no data rows")

    validated = []
    identities = set()
    for row_number, source in enumerate(source_rows, start=2):
        if not source["data_origin"].strip():
            raise ValueError(f"row {row_number}: data_origin must not be empty")
        episode_id = source["episode_id"].strip()
        cell_id = source["cell_id"].strip()
        if not episode_id or not cell_id:
            raise ValueError(f"row {row_number}: episode_id and cell_id are required")
        identity = (episode_id, cell_id)
        if identity in identities:
            raise ValueError(f"row {row_number}: duplicate episode/cell {identity}")
        identities.add(identity)

        parsed = dict(source)
        parsed["grid_row"] = _integer(source, "grid_row", row_number, 1)
        parsed["grid_column"] = _integer(source, "grid_column", row_number, 1)
        parsed["pass_count"] = _integer(source, "pass_count", row_number, 0)
        for column in (*GEOMETRY_COLUMNS, *PROCESS_COLUMNS[:-1], *SURFACE_COLUMNS,
                       *OPTICAL_COLUMNS):
            parsed[column] = _number(source, column, row_number)

        normal_length = math.sqrt(sum(parsed[name] ** 2 for name in (
            "normal_x", "normal_y", "normal_z"
        )))
        if abs(normal_length - 1.0) > normal_tolerance:
            raise ValueError(
                f"row {row_number}: surface normal length {normal_length:.6f} is not 1"
            )
        for column in PROCESS_COLUMNS:
            if parsed[column] < 0:
                raise ValueError(f"row {row_number}: {column} must be >= 0")
        if not 0.0 <= parsed["step_over_ratio"] <= 1.0:
            raise ValueError(f"row {row_number}: step_over_ratio must be in [0, 1]")
        for column in OPTICAL_COLUMNS:
            if not 0.0 <= parsed[column] <= 1.0:
                raise ValueError(f"row {row_number}: {column} must be in [0, 1]")
        for column in SURFACE_COLUMNS:
            if parsed[column] < 0:
                raise ValueError(f"row {row_number}: {column} must be >= 0")
        for column in (
            "roughness_before", "roughness_after",
            "scratch_before", "scratch_after",
        ):
            if parsed[column] > 1.0:
                raise ValueError(f"row {row_number}: {column} must be in [0, 1]")
        if parsed["clearcoat_after_um"] > parsed["clearcoat_before_um"] + clearcoat_tolerance_um:
            raise ValueError(f"row {row_number}: clearcoat increased after polishing")
        expected_removed = parsed["clearcoat_before_um"] - parsed["clearcoat_after_um"]
        if abs(expected_removed - parsed["clearcoat_removed_um"]) > clearcoat_tolerance_um:
            raise ValueError(
                f"row {row_number}: clearcoat mass balance mismatch; "
                f"before-after={expected_removed:.6f}, removed={parsed['clearcoat_removed_um']:.6f}"
            )
        validated.append(parsed)
    return validated


def evaluate(rows, config=None):
    config = config or LiteratureGuProxyConfig()
    evaluated = []
    for row in rows:
        before = relative_gloss_to_gu_proxy(row["relative_gloss_before_not_gu"], config)
        after = relative_gloss_to_gu_proxy(row["relative_gloss_after_not_gu"], config)
        result = dict(row)
        result.update({
            "predicted_20deg_gu_proxy_before": before,
            "predicted_20deg_gu_proxy_after": after,
            "predicted_20deg_gu_proxy_improvement": after - before,
            "target_gu_proxy_pass": gu_proxy_passes(after, config),
        })
        evaluated.append(result)
    return evaluated


def summarize(rows, config):
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row)
    episodes = {}
    for episode_id, episode_rows in sorted(by_episode.items()):
        before = [row["predicted_20deg_gu_proxy_before"] for row in episode_rows]
        after = [row["predicted_20deg_gu_proxy_after"] for row in episode_rows]
        passed = sum(row["target_gu_proxy_pass"] for row in episode_rows)
        episodes[episode_id] = {
            "cell_count": len(episode_rows),
            "mean_gu_proxy_before": sum(before) / len(before),
            "mean_gu_proxy_after": sum(after) / len(after),
            "mean_gu_proxy_improvement": sum(after) / len(after) - sum(before) / len(before),
            "target_pass_cell_count": passed,
            "target_pass_cell_ratio": passed / len(episode_rows),
            "minimum_clearcoat_after_um": min(row["clearcoat_after_um"] for row in episode_rows),
        }
    data_origins = sorted({row["data_origin"] for row in rows})
    return {
        "status": "validated_and_evaluated",
        "model": config.metadata(),
        "input_contract_version": "polytwin_rl_cell_output_v1",
        "row_count": len(rows),
        "data_origins": data_origins,
        "contains_only_synthetic_interface_example": all(
            origin == "synthetic_interface_example_not_rl"
            for origin in data_origins
        ),
        "episodes": episodes,
        "important": (
            "GU values are literature proxies calculated only after the RL environment "
            "provides its separate optical relative-gloss output."
        ),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(input_csv, output_dir, config=None, normal_tolerance=1e-3,
        clearcoat_tolerance_um=1e-3):
    config = config or LiteratureGuProxyConfig()
    rows = read_and_validate(input_csv, normal_tolerance, clearcoat_tolerance_um)
    evaluated = evaluate(rows, config)
    summary = summarize(evaluated, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "rl_cells_with_gu_proxy.csv"
    output_json = output_dir / "rl_gu_proxy_summary.json"
    write_csv(output_csv, evaluated)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_csv, output_json, summary


def main():
    args = parse_args()
    output_csv, output_json, summary = run(
        args.input_csv,
        args.output_dir,
        config=LiteratureGuProxyConfig(target_gu=args.target_gu),
        normal_tolerance=args.normal_tolerance,
        clearcoat_tolerance_um=args.clearcoat_tolerance_um,
    )
    print(
        f"[RL bridge] validated {summary['row_count']} rows across "
        f"{len(summary['episodes'])} episode(s)"
    )
    print(f"[RL bridge] CSV: {output_csv}")
    print(f"[RL bridge] JSON: {output_json}")


if __name__ == "__main__":
    main()
