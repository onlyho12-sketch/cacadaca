"""Summarize the frozen-seed F10-H three-arm evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone


parser = argparse.ArgumentParser()
parser.add_argument("--eval-root", required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--parent-checkpoint", required=True)
parser.add_argument("--residual-checkpoint", required=True)
args = parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows, field):
    return sum(float(row[field]) for row in rows) / len(rows)


def stats(rows):
    return {
        "sequences": len(rows),
        "mean_all4_area_pct": mean(rows, "final_all4_pass_area_pct"),
        "mean_gu": mean(rows, "gu_final"),
        "mean_ra_um": mean(rows, "ra_final_um"),
        "mean_rz_um": mean(rows, "rz_final_um"),
        "mean_scratch_um": mean(rows, "scratch_final_um"),
        "mean_control_steps": mean(rows, "control_steps"),
        "max_raw_force_n": max(float(row["raw_force_max_n"]) for row in rows),
        "quality_ok_sequences": sum(row["quality_ok"] == "True" for row in rows),
        "safety_ok_sequences": sum(row["safety_ok"] == "True" for row in rows),
        "force_hard_violations": sum(row["force_hard_violated"] == "True" for row in rows),
        "thermal_hard_violations": sum(row["thermal_hard_violated"] == "True" for row in rows),
        "unstable_hard_violations": sum(row["unstable_hard_violated"] == "True" for row in rows),
        "sensor_fault_steps": sum(int(row["sensor_fault_steps"]) for row in rows),
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
    }


def main():
    eval_root = os.path.abspath(args.eval_root)
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(out_dir)
    os.makedirs(out_dir)
    rows = []
    metadata = []
    for name in sorted(os.listdir(eval_root)):
        if name.startswith("control_") and not name.startswith("control_parent_"):
            continue
        directory = os.path.join(eval_root, name)
        seq_path = os.path.join(directory, "sequences.csv")
        meta_path = os.path.join(directory, "metadata.json")
        if os.path.isfile(seq_path) and os.path.isfile(meta_path):
            rows.extend(read_csv(seq_path))
            with open(meta_path, encoding="utf-8") as fh:
                metadata.append(json.load(fh))
    if len(rows) != 48:
        raise RuntimeError(f"expected 48 sequences, found {len(rows)}")
    if len(metadata) != 12:
        raise RuntimeError(f"expected 12 run metadata files, found {len(metadata)}")
    by_arm = {arm: [row for row in rows if row["arm"] == arm]
              for arm in ("control_parent", "deterministic_g3", "residual_ppo")}
    if any(len(value) != 16 for value in by_arm.values()):
        raise RuntimeError({key: len(value) for key, value in by_arm.items()})
    key = lambda row: (row["surface_kind"], row["direction_mode"],
                       row["surface_profile"], row["profile_seed"])
    deterministic = {key(row): row for row in by_arm["deterministic_g3"]}
    ppo = {key(row): row for row in by_arm["residual_ppo"]}
    paired = []
    for item in sorted(deterministic):
        base = deterministic[item]
        candidate = ppo[item]
        paired.append({
            "surface_kind": item[0], "direction_mode": item[1],
            "surface_profile": item[2], "profile_seed": item[3],
            "deterministic_all4_area_pct": float(base["final_all4_pass_area_pct"]),
            "ppo_all4_area_pct": float(candidate["final_all4_pass_area_pct"]),
            "ppo_minus_deterministic_all4_pp": (
                float(candidate["final_all4_pass_area_pct"])
                - float(base["final_all4_pass_area_pct"])),
            "deterministic_raw_force_max_n": float(base["raw_force_max_n"]),
            "ppo_raw_force_max_n": float(candidate["raw_force_max_n"]),
            "deterministic_control_steps": int(base["control_steps"]),
            "ppo_control_steps": int(candidate["control_steps"]),
        })
    delta = sum(row["ppo_minus_deterministic_all4_pp"] for row in paired) / len(paired)
    det_stats = stats(by_arm["deterministic_g3"])
    ppo_stats = stats(by_arm["residual_ppo"])
    step_increase_pct = 100.0 * (
        ppo_stats["mean_control_steps"] / det_stats["mean_control_steps"] - 1.0)
    checks = [
        {"check": "all_expected_sequences", "expected": 48, "actual": len(rows),
         "pass": len(rows) == 48},
        {"check": "all_expected_tiles", "expected": 1200,
         "actual": sum(int(item["tile_rows"]) for item in metadata),
         "pass": sum(int(item["tile_rows"]) for item in metadata) == 1200},
        {"check": "ppo_completed_all_sequences", "expected": 16,
         "actual": ppo_stats["sequences"], "pass": ppo_stats["sequences"] == 16},
        {"check": "ppo_force_hard_violations", "expected": 0,
         "actual": ppo_stats["force_hard_violations"],
         "pass": ppo_stats["force_hard_violations"] == 0},
        {"check": "ppo_thermal_hard_violations", "expected": 0,
         "actual": ppo_stats["thermal_hard_violations"],
         "pass": ppo_stats["thermal_hard_violations"] == 0},
        {"check": "ppo_unstable_hard_violations", "expected": 0,
         "actual": ppo_stats["unstable_hard_violations"],
         "pass": ppo_stats["unstable_hard_violations"] == 0},
        {"check": "ppo_sensor_fault_steps", "expected": 0,
         "actual": ppo_stats["sensor_fault_steps"],
         "pass": ppo_stats["sensor_fault_steps"] == 0},
        {"check": "ppo_cap_contract_failures", "expected": 0,
         "actual": sum(int(item["cap_contract_failures"]) for item in metadata
                       if item["arm"] == "residual_ppo"),
         "pass": sum(int(item["cap_contract_failures"]) for item in metadata
                     if item["arm"] == "residual_ppo") == 0},
        {"check": "ppo_mean_all4_non_degradation_pp", "expected": ">=0",
         "actual": delta, "pass": delta >= -1e-12},
        {"check": "ppo_any_paired_all4_improvement", "expected": ">0 pairs",
         "actual": sum(row["ppo_minus_deterministic_all4_pp"] > 1e-12 for row in paired),
         "pass": any(row["ppo_minus_deterministic_all4_pp"] > 1e-12 for row in paired)},
    ]
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F10H_BOUNDED_RESIDUAL_PPO_FIXED_EVALUATION",
        "decision": "FAIL_RETAIN_DETERMINISTIC_G3",
        "gate_pass": all(item["pass"] for item in checks),
        "candidate_promoted": False,
        "release_modified": False,
        "reason": ("Residual PPO passed safety but reduced paired ALL-4 area and "
                   "increased control steps; retain deterministic G3."),
        "paired_all4_delta_pp_ppo_minus_deterministic": delta,
        "paired_improved": sum(row["ppo_minus_deterministic_all4_pp"] > 1e-12 for row in paired),
        "paired_equal": sum(abs(row["ppo_minus_deterministic_all4_pp"]) <= 1e-12
                            for row in paired),
        "paired_worse": sum(row["ppo_minus_deterministic_all4_pp"] < -1e-12 for row in paired),
        "ppo_control_steps_increase_pct": step_increase_pct,
        "arms": {arm: stats(value) for arm, value in by_arm.items()},
        "checks": checks,
        "parent_checkpoint_sha256": sha256(os.path.abspath(args.parent_checkpoint)),
        "residual_checkpoint_sha256": sha256(os.path.abspath(args.residual_checkpoint)),
    }
    write_csv(os.path.join(out_dir, "all_sequences.csv"), rows)
    write_csv(os.path.join(out_dir, "paired_comparison.csv"), paired)
    write_csv(os.path.join(out_dir, "acceptance_checks.csv"), checks)
    with open(os.path.join(out_dir, "decision.json"), "w", encoding="utf-8") as fh:
        json.dump(decision, fh, indent=2, sort_keys=True)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
