"""Aggregate Gate F6 zero-shot geometry runs against pre-frozen criteria."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from typing import Any

import numpy as np


GEOMETRIES = ("flat", "cylinder", "sphere", "freeform")
EXPECTED_DIRECTIONS = ("same_xx", "cross_xy")
EXPECTED_PROFILES = (
    "legacy_stress", "factory_prepolish", "factory_prepolish_deep_defect",
    "factory_prepolish_deep_stress",
)
CRITERIA = {
    "completed_sequences": 64,
    "tile_rows": 1600,
    "sensor_fault_steps": 0,
    "curved_safety_rate_min": 1.0,
    "quality_pass_count_drop_max": 0,
    "all4_cell_area_drop_max_percentage_points": 1.0,
    "mean_control_steps_increase_max_pct": 40.0,
}


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _truth(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def _key(row: dict[str, str]) -> tuple[str, str, int]:
    return row["direction_mode"], row["surface_profile"], int(row["profile_seed"])


def _all4_area(tile_rows: list[dict[str, str]]) -> tuple[int, int, float]:
    total = sum(int(row["cell_count"]) for row in tile_rows)
    passed = sum(int(round(
        int(row["cell_count"]) * float(row["all4_pass_area_pct"]) / 100.0
    )) for row in tile_rows)
    return passed, total, 100.0 * passed / total


def summarize_geometry(
    kind: str, sequences: list[dict[str, str]], tiles: list[dict[str, str]]
) -> dict[str, Any]:
    passed_cells, total_cells, area = _all4_area(tiles)
    tile_area = np.asarray([float(row["all4_pass_area_pct"]) for row in tiles])
    outcomes: dict[str, int] = {}
    for row in sequences:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
    return {
        "surface_kind": kind,
        "sequences": len(sequences),
        "safety_pass": sum(_truth(row["safety_ok"]) for row in sequences),
        "quality_pass": sum(_truth(row["quality_ok"]) for row in sequences),
        "sensor_fault_steps": sum(int(row["sensor_fault_steps"]) for row in sequences),
        "force_overload_failures": outcomes.get("fail_force_overload", 0),
        "max_pass_failures": outcomes.get("fail_max_passes", 0),
        "mean_control_steps": float(np.mean([
            int(row["control_steps"]) for row in sequences])),
        "mean_force_action": float(np.mean([
            float(row["force_action_mean"]) for row in sequences])),
        "mean_feed_action": float(np.mean([
            float(row["feed_action_mean"]) for row in sequences])),
        "all4_cells_pass": passed_cells,
        "surface_cells_total": total_cells,
        "all4_cell_area_pct": area,
        "tiles_total": len(tiles),
        "tiles_all4_area_100pct": int(np.sum(tile_area >= 100.0 - 1.0e-9)),
        "tiles_all4_area_ge95pct": int(np.sum(tile_area >= 95.0)),
        "tiles_all4_area_ge90pct": int(np.sum(tile_area >= 90.0)),
        "tiles_all4_area_ge80pct": int(np.sum(tile_area >= 80.0)),
        "outcomes_json": json.dumps(outcomes, sort_keys=True),
    }


def aggregate(input_dirs: list[str], out_dir: str) -> dict[str, Any]:
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    sequences: list[dict[str, str]] = []
    tiles: list[dict[str, str]] = []
    metadata = []
    for directory in input_dirs:
        sequences.extend(_read_csv(os.path.join(directory, "sequences.csv")))
        tiles.extend(_read_csv(os.path.join(directory, "tile_area_fractions.csv")))
        with open(os.path.join(directory, "metadata.json"), encoding="utf-8") as handle:
            metadata.append(json.load(handle))
    kinds = {row["surface_kind"] for row in sequences}
    if kinds != set(GEOMETRIES):
        raise ValueError(f"expected geometries {GEOMETRIES}, got {sorted(kinds)}")
    expected_keys = {
        (direction, profile, seed)
        for direction in EXPECTED_DIRECTIONS
        for profile in EXPECTED_PROFILES
        for seed in (31000, 31097)
    }
    by_geometry: dict[str, dict[tuple[str, str, int], dict[str, str]]] = {}
    geometry_rows = []
    for kind in GEOMETRIES:
        seq = [row for row in sequences if row["surface_kind"] == kind]
        tile = [row for row in tiles if row["surface_kind"] == kind]
        keyed = {_key(row): row for row in seq}
        if len(keyed) != len(seq) or set(keyed) != expected_keys:
            raise ValueError(f"{kind} paired keys incomplete or duplicated")
        if len(tile) != 400:
            raise ValueError(f"{kind} expected 400 tiles, got {len(tile)}")
        by_geometry[kind] = keyed
        geometry_rows.append(summarize_geometry(kind, seq, tile))
    summary_by_kind = {row["surface_kind"]: row for row in geometry_rows}
    flat = summary_by_kind["flat"]

    paired_rows = []
    for kind in GEOMETRIES[1:]:
        curved = summary_by_kind[kind]
        step_delta_pct = 100.0 * (
            curved["mean_control_steps"] / flat["mean_control_steps"] - 1.0)
        paired_rows.append({
            "surface_kind": kind,
            "safety_pass": curved["safety_pass"],
            "required_safety_pass": curved["sequences"],
            "quality_pass": curved["quality_pass"],
            "flat_quality_pass": flat["quality_pass"],
            "quality_pass_count_delta": curved["quality_pass"] - flat["quality_pass"],
            "all4_cell_area_pct": curved["all4_cell_area_pct"],
            "flat_all4_cell_area_pct": flat["all4_cell_area_pct"],
            "all4_delta_percentage_points": (
                curved["all4_cell_area_pct"] - flat["all4_cell_area_pct"]),
            "mean_control_steps": curved["mean_control_steps"],
            "flat_mean_control_steps": flat["mean_control_steps"],
            "mean_control_steps_delta_pct": step_delta_pct,
        })

    check_rows = []
    def check(scope: str, name: str, value: Any, operator: str, threshold: Any) -> None:
        passed = {
            "eq": value == threshold,
            "ge": value >= threshold,
            "le": value <= threshold,
        }[operator]
        check_rows.append({
            "scope": scope, "check": name, "observed": value,
            "operator": operator, "threshold": threshold, "passed": passed,
        })

    check("completion", "completed_sequences", len(sequences), "eq", 64)
    check("completion", "tile_rows", len(tiles), "eq", 1600)
    check("completion", "sensor_fault_steps", sum(
        int(row["sensor_fault_steps"]) for row in sequences), "eq", 0)
    finite = all(
        np.isfinite(float(row[key]))
        for row in sequences
        for key in ("control_steps", "force_action_mean", "feed_action_mean",
                    "final_all4_pass_area_pct")
    )
    check("completion", "selected_numeric_outputs_finite", finite, "eq", True)
    for row in paired_rows:
        kind = row["surface_kind"]
        check(kind, "all_sequences_safety_ok", row["safety_pass"], "eq", 16)
        check(kind, "quality_pass_count_delta", row["quality_pass_count_delta"], "ge", 0)
        check(kind, "all4_delta_percentage_points", row[
            "all4_delta_percentage_points"], "ge", -1.0)
        check(kind, "mean_control_steps_delta_pct", row[
            "mean_control_steps_delta_pct"], "le", 40.0)

    failures = []
    for row in sequences:
        if row["outcome"] != "success" or not _truth(row["safety_ok"]):
            failures.append({
                "surface_kind": row["surface_kind"],
                "direction_mode": row["direction_mode"],
                "surface_profile": row["surface_profile"],
                "profile_seed": row["profile_seed"],
                "outcome": row["outcome"],
                "safety_ok": row["safety_ok"],
                "quality_ok": row["quality_ok"],
                "control_steps": row["control_steps"],
                "sensor_fault_steps": row["sensor_fault_steps"],
                "force_hard_violated": row["force_hard_violated"],
                "thermal_hard_violated": row["thermal_hard_violated"],
                "unstable_hard_violated": row["unstable_hard_violated"],
                "final_all4_pass_area_pct": row["final_all4_pass_area_pct"],
            })
    profile_rows = []
    for kind in GEOMETRIES:
        for profile in EXPECTED_PROFILES:
            group = [row for row in sequences
                     if row["surface_kind"] == kind and row["surface_profile"] == profile]
            profile_rows.append({
                "surface_kind": kind,
                "surface_profile": profile,
                "sequences": len(group),
                "safety_pass": sum(_truth(row["safety_ok"]) for row in group),
                "quality_pass": sum(_truth(row["quality_ok"]) for row in group),
                "force_overload_failures": sum(
                    row["outcome"] == "fail_force_overload" for row in group),
                "mean_control_steps": float(np.mean([
                    int(row["control_steps"]) for row in group])),
                "mean_final_all4_area_pct": float(np.mean([
                    float(row["final_all4_pass_area_pct"]) for row in group])),
            })

    completion_pass = all(row["passed"] for row in check_rows if row["scope"] == "completion")
    sufficiency_pass = all(row["passed"] for row in check_rows if row["scope"] != "completion")
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F6",
        "completion_decision": "PASS" if completion_pass else "FAIL",
        "zero_shot_sufficiency_decision": "PASS" if sufficiency_pass else "FAIL",
        "completed_sequences": len(sequences),
        "tile_rows": len(tiles),
        "failure_rows": len(failures),
        "training_performed": False,
        "criteria_frozen_before_physx": CRITERIA,
        "next_step": (
            "Zero-shot is sufficient; stop and request approval before further validation."
            if sufficiency_pass else
            "Zero-shot is insufficient; F7 observation/action pilot requires separate user approval."
        ),
    }
    os.makedirs(out_dir)
    _write_csv(os.path.join(out_dir, "combined_sequences.csv"), sequences)
    _write_csv(os.path.join(out_dir, "combined_tile_area_fractions.csv"), tiles)
    _write_csv(os.path.join(out_dir, "geometry_summary.csv"), geometry_rows)
    _write_csv(os.path.join(out_dir, "profile_geometry_summary.csv"), profile_rows)
    _write_csv(os.path.join(out_dir, "paired_vs_flat.csv"), paired_rows)
    _write_csv(os.path.join(out_dir, "acceptance_checks.csv"), check_rows)
    if failures:
        _write_csv(os.path.join(out_dir, "geometry_failures.csv"), failures)
    else:
        with open(os.path.join(out_dir, "geometry_failures.csv"), "w", encoding="utf-8") as handle:
            handle.write("surface_kind,direction_mode,surface_profile,profile_seed,outcome\n")
    with open(os.path.join(out_dir, "decision.json"), "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.input_dir, args.out_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
