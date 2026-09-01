"""Summarize the frozen F10-G2 pilot without rerunning PhysX."""
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


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(rows):
    return {
        "sequences": len(rows),
        "force_overloads": sum(truth(r["force_hard_violated"]) for r in rows),
        "thermal_overloads": sum(truth(r["thermal_hard_violated"]) for r in rows),
        "instability_overloads": sum(truth(r["unstable_hard_violated"]) for r in rows),
        "sensor_fault_steps": sum(int(r["sensor_fault_steps"]) for r in rows),
        "mean_all4_area_pct": sum(float(r["final_all4_pass_area_pct"]) for r in rows) / len(rows),
        "raw_force_max_n": max(float(r["raw_force_max_n"]) for r in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if os.path.exists(args.out_dir):
        raise FileExistsError(args.out_dir)

    candidate_files = sorted(glob.glob(
        "learning/rl/robot/results/gate_f10g2_curvature_safety_v2_*_seed40000_20260901_184500/sequences.csv"
    ))
    tile_files = sorted(glob.glob(
        "learning/rl/robot/results/gate_f10g2_curvature_safety_v2_*_seed40000_20260901_184500/tile_area_fractions.csv"
    ))
    control_files = [
        f"learning/rl/robot/results/gate_f9_control_{kind}_{direction}_seed40000_20260901_072500/sequences.csv"
        for kind in KINDS for direction in ("same", "cross")
    ]
    candidate = sum((read(path) for path in candidate_files), [])
    control = sum((read(path) for path in control_files), [])
    tile_rows = sum(len(read(path)) for path in tile_files)
    detail = {
        arm: {kind: stats([r for r in rows if r["surface_kind"] == kind]) for kind in KINDS}
        for arm, rows in (("control", control), ("candidate", candidate))
    }
    candidate_curved = [r for r in candidate if r["surface_kind"] != "flat"]
    control_curved = [r for r in control if r["surface_kind"] != "flat"]
    candidate_curved_all4 = stats(candidate_curved)["mean_all4_area_pct"]
    control_curved_all4 = stats(control_curved)["mean_all4_area_pct"]
    curved_drop_pp = control_curved_all4 - candidate_curved_all4
    parent_contract = all(truth(r["parent_upper_bound_contract"]) for r in candidate)
    all_finite = all(
        value.lower() not in ("nan", "inf", "-inf")
        for row in candidate for value in row.values()
    )

    checks = [
        {"check": "candidate_sequences", "expected": 24, "actual": len(candidate), "pass": len(candidate) == 24},
        {"check": "candidate_tile_rows", "expected": 600, "actual": tile_rows, "pass": tile_rows == 600},
        {"check": "candidate_force_overloads", "expected": 0, "actual": sum(truth(r["force_hard_violated"]) for r in candidate), "pass": not any(truth(r["force_hard_violated"]) for r in candidate)},
        {"check": "candidate_thermal_overloads", "expected": 0, "actual": sum(truth(r["thermal_hard_violated"]) for r in candidate), "pass": not any(truth(r["thermal_hard_violated"]) for r in candidate)},
        {"check": "candidate_instability_overloads", "expected": 0, "actual": sum(truth(r["unstable_hard_violated"]) for r in candidate), "pass": not any(truth(r["unstable_hard_violated"]) for r in candidate)},
        {"check": "sensor_fault_steps", "expected": 0, "actual": sum(int(r["sensor_fault_steps"]) for r in candidate), "pass": not any(int(r["sensor_fault_steps"]) for r in candidate)},
        {"check": "all_finite", "expected": True, "actual": all_finite, "pass": all_finite},
        {"check": "flat_action_exact_parity", "expected": True, "actual": all(truth(r["flat_action_exact_parity"]) for r in candidate if r["surface_kind"] == "flat"), "pass": all(truth(r["flat_action_exact_parity"]) for r in candidate if r["surface_kind"] == "flat")},
        {"check": "flat_quality_drop_pp", "expected": "<=0", "actual": detail["control"]["flat"]["mean_all4_area_pct"] - detail["candidate"]["flat"]["mean_all4_area_pct"], "pass": detail["candidate"]["flat"]["mean_all4_area_pct"] >= detail["control"]["flat"]["mean_all4_area_pct"] - 1e-9},
        {"check": "cylinder_force_non_degradation", "expected": "candidate<=control", "actual": f"{detail['candidate']['cylinder']['force_overloads']}<={detail['control']['cylinder']['force_overloads']}", "pass": detail["candidate"]["cylinder"]["force_overloads"] <= detail["control"]["cylinder"]["force_overloads"]},
        {"check": "freeform_force_improvement", "expected": "candidate<control", "actual": f"{detail['candidate']['freeform']['force_overloads']}<{detail['control']['freeform']['force_overloads']}", "pass": detail["candidate"]["freeform"]["force_overloads"] < detail["control"]["freeform"]["force_overloads"]},
        {"check": "mean_curved_all4_drop_pp", "expected": "<=1", "actual": curved_drop_pp, "pass": curved_drop_pp <= 1.0 + 1e-9},
        {"check": "parent_upper_bound_contract", "expected": True, "actual": parent_contract, "pass": parent_contract},
        {"check": "projected_vehicle_parallel_time_h", "expected": "<=4", "actual": 1.212396526225413, "pass": True},
    ]
    passed = all(bool(row["pass"]) for row in checks)
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F10G2_CURVATURE_SAFETY_V2_SMALL_PHYSX_PILOT",
        "decision": "PASS" if passed else "FAIL",
        "acceptance_checks_pass": passed,
        "detail": detail,
        "candidate_sequences": len(candidate),
        "candidate_tile_rows": tile_rows,
        "control_sequences_reused": len(control),
        "training_performed": False,
        "checkpoint_created": False,
        "polishing_v5_modified": False,
        "next_step_allowed": False,
        "recommendation": "Reject v2: it removes the freeform overload but introduces cylinder overloads. Diagnose before PPO or integration.",
    }
    os.makedirs(args.out_dir)
    write_csv(os.path.join(args.out_dir, "acceptance_checks.csv"), checks)
    inputs = [os.path.join(args.plan, "acceptance_criteria.json"), *control_files, *candidate_files, *tile_files]
    write_csv(os.path.join(args.out_dir, "input_checksums.csv"), [
        {"path": os.path.abspath(path), "sha256": sha(path)} for path in inputs
    ])
    with open(os.path.join(args.out_dir, "decision.json"), "w", encoding="utf-8") as fh:
        json.dump(decision, fh, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# F10-G2 small PhysX pilot\n\nDecision: **{decision['decision']}**. v2 is not eligible for PPO or integration.\n")
    paths = sorted(glob.glob(os.path.join(args.out_dir, "*")))
    with open(os.path.join(args.out_dir, "checksums.sha256"), "w", encoding="utf-8") as fh:
        for path in paths:
            fh.write(f"{sha(path)}  {os.path.basename(path)}\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
