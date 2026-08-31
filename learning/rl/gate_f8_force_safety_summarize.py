"""Aggregate Gate F8 force-safety pilot arms against pre-frozen criteria.

Stage 1 applies the frozen stage2-entry rule; stage `final` applies the frozen
final gate to control plus the selected candidate across both directions.  No
threshold in this file may be edited after PhysX results exist.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from typing import Any

import numpy as np


GEOMETRIES = ("flat", "cylinder", "sphere", "freeform")
CURVED = ("cylinder", "sphere", "freeform")
ENTRY_CURVED = ("cylinder", "freeform")
PROFILES = (
    "legacy_stress", "factory_prepolish", "factory_prepolish_deep_defect",
    "factory_prepolish_deep_stress",
)
CONTROL = "control"
CANDIDATES = ("static_cap", "predictive_shield")
PARENT_SHA256 = "c733763bfebd7fcead01cf2574bbd1621c4660b9347143cccb35e0e83449d052"
FIXED_SEEDS = {"physics_seed": 20260901, "policy_seed": 20260831, "surface_seed_base": 33000}
FLAT_PARITY_FIELDS = (
    "outcome", "safety_ok", "quality_ok", "control_steps", "contact_steps",
    "sensor_fault_steps", "raw_force_max_n", "gu_final", "ra_final_um", "rz_final_um",
    "scratch_final_um", "temperature_peak_c", "final_all4_pass_area_pct",
)
SEQUENCES_PER_ARM_DIRECTION = 16
TILES_PER_ARM_DIRECTION = 400
CRITERIA = {
    "stage2_entry": {
        "entry_curved_force_overloads_below_control": True,
        "sensor_fault_steps": 0,
        "flat_safety_pass_not_below_control": True,
        "flat_action_passthrough_parity": True,
    },
    "final_gate": {
        "entry_curved_safety_pass_gt_control": True,
        "candidate_force_overloads_both_directions": 0,
        "flat_safety_pass_not_below_control": True,
        "flat_quality_pass_not_below_control": True,
        "mean_curved_all4_area_drop_max_percentage_points": 1.0,
        "sensor_fault_steps": 0,
    },
    "candidate_selection_tiebreak": [
        "fewer cylinder+freeform force overloads",
        "higher curved safety pass count",
        "smaller paired curved all4 area drop versus control",
        "less intrusive shield (static_cap before predictive_shield)",
    ],
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


def _key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["surface_kind"], row["direction_mode"], row["surface_profile"],
            row["profile_seed"])


def load_runs(input_dirs: list[str]) -> tuple[list[dict[str, str]], list[dict[str, str]],
                                              list[dict[str, Any]]]:
    sequences: list[dict[str, str]] = []
    tiles: list[dict[str, str]] = []
    metadata: list[dict[str, Any]] = []
    for directory in input_dirs:
        with open(os.path.join(directory, "metadata.json"), encoding="utf-8") as handle:
            meta = json.load(handle)
        if meta.get("gate") != "F8_FORCE_SAFETY":
            raise ValueError(f"{directory}: not a Gate F8 evaluation run")
        if meta.get("training_performed") is not False:
            raise ValueError(f"{directory}: training_performed must be false")
        if meta.get("observation_mode") != "base14":
            raise ValueError(f"{directory}: observation_mode must be base14")
        if meta.get("checkpoint_sha256") != PARENT_SHA256:
            raise ValueError(f"{directory}: unexpected parent checkpoint hash")
        if meta.get("all_finite") is not True:
            raise ValueError(f"{directory}: metadata reports non-finite outputs")
        if int(meta.get("completed", -1)) != int(meta.get("expected", -2)):
            raise ValueError(f"{directory}: incomplete run")
        for field, expected in FIXED_SEEDS.items():
            if int(meta.get(field, -1)) != expected:
                raise ValueError(f"{directory}: {field} must be {expected}")
        metadata.append(meta)
        sequences.extend(_read_csv(os.path.join(directory, "sequences.csv")))
        tiles.extend(_read_csv(os.path.join(directory, "tile_area_fractions.csv")))
    return sequences, tiles, metadata


def _validate_arms(sequences: list[dict[str, str]], tiles: list[dict[str, str]],
                   arms: tuple[str, ...], directions: tuple[str, ...]) -> None:
    expected_keys = {
        (kind, direction, profile, str(FIXED_SEEDS["surface_seed_base"]))
        for kind in GEOMETRIES for direction in directions for profile in PROFILES
    }
    for arm in arms:
        for direction in directions:
            rows = [row for row in sequences
                    if row["shield_mode"] == arm and row["direction_mode"] == direction]
            keyed = {_key(row): row for row in rows}
            if len(rows) != SEQUENCES_PER_ARM_DIRECTION or len(keyed) != len(rows):
                raise ValueError(f"{arm}/{direction}: expected "
                                 f"{SEQUENCES_PER_ARM_DIRECTION} unique sequences, got {len(rows)}")
            direction_keys = {key for key in expected_keys if key[1] == direction}
            if set(keyed) != direction_keys:
                raise ValueError(f"{arm}/{direction}: paired keys incomplete")
            tile_rows = [row for row in tiles
                         if row["shield_mode"] == arm and row["direction_mode"] == direction]
            if len(tile_rows) != TILES_PER_ARM_DIRECTION:
                raise ValueError(f"{arm}/{direction}: expected {TILES_PER_ARM_DIRECTION} "
                                 f"tile rows, got {len(tile_rows)}")


def _subset(sequences: list[dict[str, str]], arm: str, kinds: tuple[str, ...] | None = None,
            directions: tuple[str, ...] | None = None) -> list[dict[str, str]]:
    return [row for row in sequences
            if row["shield_mode"] == arm
            and (kinds is None or row["surface_kind"] in kinds)
            and (directions is None or row["direction_mode"] in directions)]


def _stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "sequences": len(rows),
        "safety_pass": sum(_truth(row["safety_ok"]) for row in rows),
        "quality_pass": sum(_truth(row["quality_ok"]) for row in rows),
        "force_overloads": sum(_truth(row["force_hard_violated"]) for row in rows),
        "thermal_overloads": sum(_truth(row["thermal_hard_violated"]) for row in rows),
        "unstable_overloads": sum(_truth(row["unstable_hard_violated"]) for row in rows),
        "sensor_fault_steps": sum(int(row["sensor_fault_steps"]) for row in rows),
        "mean_control_steps": float(np.mean([int(row["control_steps"]) for row in rows])),
        "max_raw_force_n": float(max(float(row["raw_force_max_n"]) for row in rows)),
        "mean_final_all4_area_pct": float(np.mean([
            float(row["final_all4_pass_area_pct"]) for row in rows])),
        "mean_parent_force_action": float(np.mean([
            float(row["parent_force_action_mean"]) for row in rows])),
        "mean_executed_force_action": float(np.mean([
            float(row["executed_force_action_mean"]) for row in rows])),
        "mean_shield_step_fraction": float(np.mean([
            float(row["shield_step_fraction"]) for row in rows])),
        "min_force_action_cap": float(min(float(row["force_action_cap_min"]) for row in rows)),
    }


def geometry_rows(sequences: list[dict[str, str]], arms: tuple[str, ...],
                  directions: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for arm in arms:
        for direction in directions:
            for kind in GEOMETRIES:
                group = _subset(sequences, arm, (kind,), (direction,))
                rows.append({"shield_mode": arm, "direction_mode": direction,
                             "surface_kind": kind, **_stats(group)})
    return rows


def flat_parity_rows(sequences: list[dict[str, str]], candidate: str,
                     directions: tuple[str, ...]) -> list[dict[str, Any]]:
    control = {_key(row): row for row in _subset(sequences, CONTROL, ("flat",), directions)}
    rows = []
    for row in _subset(sequences, candidate, ("flat",), directions):
        reference = control[_key(row)]
        mismatched = [field for field in FLAT_PARITY_FIELDS if row[field] != reference[field]]
        rows.append({
            "shield_mode": candidate, "direction_mode": row["direction_mode"],
            "surface_profile": row["surface_profile"], "profile_seed": row["profile_seed"],
            "mismatched_fields": ";".join(mismatched), "identical": not mismatched,
        })
    return rows


def paired_all4_drop_pp(sequences: list[dict[str, str]], candidate: str,
                        kinds: tuple[str, ...], directions: tuple[str, ...]) -> float:
    control = {_key(row): row for row in _subset(sequences, CONTROL, kinds, directions)}
    drops = [float(control[_key(row)]["final_all4_pass_area_pct"])
             - float(row["final_all4_pass_area_pct"])
             for row in _subset(sequences, candidate, kinds, directions)]
    return float(np.mean(drops))


class _Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, scope: str, name: str, value: Any, operator: str, threshold: Any) -> bool:
        passed = {
            "eq": lambda: value == threshold,
            "ge": lambda: value >= threshold,
            "gt": lambda: value > threshold,
            "le": lambda: value <= threshold,
            "lt": lambda: value < threshold,
        }[operator]()
        self.rows.append({"scope": scope, "check": name, "observed": value,
                          "operator": operator, "threshold": threshold, "passed": passed})
        return passed

    def passed(self, scope: str | None = None) -> bool:
        return all(row["passed"] for row in self.rows
                   if scope is None or row["scope"] == scope)


def evaluate_stage1(sequences: list[dict[str, str]]) -> tuple[_Checks, dict[str, Any]]:
    checks = _Checks()
    directions = ("same_xx",)
    control_entry = _stats(_subset(sequences, CONTROL, ENTRY_CURVED, directions))
    control_flat = _stats(_subset(sequences, CONTROL, ("flat",), directions))
    checks.add("completion", "sequences", len(sequences), "eq",
               SEQUENCES_PER_ARM_DIRECTION * (1 + len(CANDIDATES)))
    entered: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        entry = _stats(_subset(sequences, candidate, ENTRY_CURVED, directions))
        flat = _stats(_subset(sequences, candidate, ("flat",), directions))
        curved = _stats(_subset(sequences, candidate, CURVED, directions))
        parity = flat_parity_rows(sequences, candidate, directions)
        overload_ok = checks.add(candidate, "entry_curved_force_overloads", entry[
            "force_overloads"], "lt", control_entry["force_overloads"])
        fault_ok = checks.add(candidate, "sensor_fault_steps", _stats(
            _subset(sequences, candidate, None, directions))["sensor_fault_steps"], "eq", 0)
        flat_ok = checks.add(candidate, "flat_safety_pass", flat["safety_pass"], "ge",
                             control_flat["safety_pass"])
        parity_ok = checks.add(candidate, "flat_action_passthrough_parity",
                               all(row["identical"] for row in parity), "eq", True)
        if overload_ok and fault_ok and flat_ok and parity_ok:
            entered.append({
                "shield_mode": candidate,
                "entry_curved_force_overloads": entry["force_overloads"],
                "curved_safety_pass": curved["safety_pass"],
                "curved_all4_drop_pp": paired_all4_drop_pp(
                    sequences, candidate, CURVED, directions),
                "intrusiveness_rank": CANDIDATES.index(candidate),
            })
    selected = None
    if entered:
        selected = sorted(entered, key=lambda row: (
            row["entry_curved_force_overloads"], -row["curved_safety_pass"],
            row["curved_all4_drop_pp"], row["intrusiveness_rank"]))[0]["shield_mode"]
    return checks, {"entered_candidates": [row["shield_mode"] for row in entered],
                    "selection_table": entered, "selected_candidate": selected,
                    "control_entry_force_overloads": control_entry["force_overloads"]}


def evaluate_final(sequences: list[dict[str, str]], candidate: str
                   ) -> tuple[_Checks, dict[str, Any]]:
    checks = _Checks()
    directions = ("same_xx", "cross_xy")
    control_entry = _stats(_subset(sequences, CONTROL, ENTRY_CURVED, directions))
    control_flat = _stats(_subset(sequences, CONTROL, ("flat",), directions))
    entry = _stats(_subset(sequences, candidate, ENTRY_CURVED, directions))
    flat = _stats(_subset(sequences, candidate, ("flat",), directions))
    all_rows = _stats(_subset(sequences, candidate, None, directions))
    drop = paired_all4_drop_pp(sequences, candidate, CURVED, directions)
    parity = flat_parity_rows(sequences, candidate, directions)
    checks.add("completion", "sequences", len(sequences), "eq",
               2 * len(directions) * SEQUENCES_PER_ARM_DIRECTION)
    checks.add("final", "entry_curved_safety_pass", entry["safety_pass"], "gt",
               control_entry["safety_pass"])
    checks.add("final", "candidate_force_overloads", all_rows["force_overloads"], "eq", 0)
    checks.add("final", "flat_safety_pass", flat["safety_pass"], "ge", control_flat["safety_pass"])
    checks.add("final", "flat_quality_pass", flat["quality_pass"], "ge",
               control_flat["quality_pass"])
    checks.add("final", "mean_curved_all4_area_drop_pp", drop, "le", 1.0)
    checks.add("final", "sensor_fault_steps", all_rows["sensor_fault_steps"], "eq", 0)
    checks.add("final", "flat_action_passthrough_parity",
               all(row["identical"] for row in parity), "eq", True)
    return checks, {
        "candidate": candidate,
        "candidate_entry_curved_safety_pass": entry["safety_pass"],
        "control_entry_curved_safety_pass": control_entry["safety_pass"],
        "candidate_force_overloads": all_rows["force_overloads"],
        "control_force_overloads": _stats(
            _subset(sequences, CONTROL, None, directions))["force_overloads"],
        "mean_curved_all4_area_drop_pp": drop,
    }


def aggregate(input_dirs: list[str], out_dir: str, stage: str,
              candidate: str | None = None) -> dict[str, Any]:
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    sequences, tiles, metadata = load_runs(input_dirs)
    if stage == "1":
        arms, directions = (CONTROL,) + CANDIDATES, ("same_xx",)
    else:
        if candidate not in CANDIDATES:
            raise ValueError("final stage requires --candidate")
        arms, directions = (CONTROL, candidate), ("same_xx", "cross_xy")
    _validate_arms(sequences, tiles, arms, directions)
    sequences = [row for row in sequences if row["shield_mode"] in arms
                 and row["direction_mode"] in directions]
    tiles = [row for row in tiles if row["shield_mode"] in arms
             and row["direction_mode"] in directions]
    checks, detail = (evaluate_stage1(sequences) if stage == "1"
                      else evaluate_final(sequences, candidate))
    geometry = geometry_rows(sequences, arms, directions)
    parity = [row for arm in arms if arm != CONTROL
              for row in flat_parity_rows(sequences, arm, directions)]
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F8_SMALL_FORCE_SAFETY_CONSTRAINT_PILOT",
        "stage": stage,
        "decision": "PASS" if checks.passed() else "FAIL",
        "completion_pass": checks.passed("completion"),
        "runs": len(metadata),
        "sequences": len(sequences),
        "tile_rows": len(tiles),
        "criteria_frozen_before_physx": CRITERIA,
        "parent_sha256": PARENT_SHA256,
        "training_performed": False,
        "champion_promoted": False,
        **detail,
    }
    os.makedirs(out_dir)
    _write_csv(os.path.join(out_dir, "combined_sequences.csv"), sequences)
    _write_csv(os.path.join(out_dir, "combined_tile_area_fractions.csv"), tiles)
    _write_csv(os.path.join(out_dir, "arm_geometry_summary.csv"), geometry)
    _write_csv(os.path.join(out_dir, "flat_passthrough_parity.csv"), parity)
    _write_csv(os.path.join(out_dir, "acceptance_checks.csv"), checks.rows)
    with open(os.path.join(out_dir, "decision.json"), "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stage", choices=("1", "final"), required=True)
    parser.add_argument("--candidate", choices=CANDIDATES)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.input_dir, args.out_dir, args.stage, args.candidate),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
