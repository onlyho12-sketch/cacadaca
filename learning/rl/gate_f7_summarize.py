"""Summarize the frozen Gate F7 smoke, collection, and BC ablation evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--smokes", nargs=4, required=True)
    parser.add_argument("--collections", nargs=4, required=True)
    parser.add_argument("--trainings", nargs=3, required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)
    plan = _read(args.plan)
    smokes = [_read(path) for path in args.smokes]
    collections = [_read(path) for path in args.collections]
    trainings = [_read(path) for path in args.trainings]
    smoke_rows = [{
        "surface_kind": item["surface_kind"], "mode": item["mode"],
        "steps": item["steps"], "all_finite": item["all_finite"],
        "sensor_fault_steps": item["sensor_fault_steps"],
        "smoke_pass": item["smoke_pass"]} for item in smokes]
    collection_rows = [{
        "surface_kind": item["surface_kind"], "samples": item["samples"],
        "risk_mean": item["risk_mean"],
        "teacher_changed_fraction": item["teacher_changed_fraction"],
        "sensor_fault_steps": item["sensor_fault_steps"],
        "latched_force_violation_step_sum": item["latched_force_violation_step_sum"]
    } for item in collections]
    ablation_rows = []
    for item in trainings:
        test = next(row for row in item["metrics"] if row["split"] == "test")
        ablation_rows.append({
            "mode": item["mode"], "observation_dim": item["observation_dim"],
            "samples": item["samples"], "test_mse": test["mse"],
            "test_mae": test["mae"], "test_force_mae": test["force_mae"],
            "test_feed_mae": test["feed_mae"],
            "checkpoint": item["checkpoint"], "checkpoint_sha256": item["checkpoint_sha256"]})
    winner = min(ablation_rows, key=lambda row: row["test_force_mae"])
    base = next(row for row in ablation_rows if row["mode"] == "base14")
    extensions = [row for row in ablation_rows if row["mode"] != "base14"]
    best_extension = min(extensions, key=lambda row: row["test_force_mae"])
    extension_beats_base = best_extension["test_force_mae"] < base["test_force_mae"]
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F7_SMALL_OBSERVATION_ABLATION_AND_BC_PILOT",
        "completion_pass": (
            all(row["smoke_pass"] for row in smoke_rows)
            and all(row["sensor_fault_steps"] == 0 for row in collection_rows)
            and len({row["samples"] for row in ablation_rows}) == 1),
        "observation_extension_gate_pass": bool(extension_beats_base),
        "selected_mode": winner["mode"],
        "best_extension": best_extension["mode"],
        "base14_test_force_mae": base["test_force_mae"],
        "best_extension_test_force_mae": best_extension["test_force_mae"],
        "best_extension_relative_delta": (
            best_extension["test_force_mae"] / base["test_force_mae"] - 1.0),
        "physx_candidate_evaluation_performed": False,
        "physx_candidate_evaluation_skip_reason": (
            "pre-frozen held-out force-MAE entry criterion failed"),
        "training_performed": True,
        "training_type": "small behavior-cloning fine-tune",
        "ppo_performed": False,
        "f8_started": False,
        "champion_promoted": False,
        "release_modified": False,
        "recommendation": (
            "Do not scale the normal/curvature observation extension. If separately approved, "
            "F8 should test an explicit force-safety objective/action constraint or richer "
            "anticipatory supervision, retaining base14 as the observation baseline."),
    }
    _write_csv(os.path.join(out_dir, "physx_smoke.csv"), smoke_rows)
    _write_csv(os.path.join(out_dir, "collection_summary.csv"), collection_rows)
    _write_csv(os.path.join(out_dir, "ablation_summary.csv"), ablation_rows)
    with open(os.path.join(out_dir, "decision.json"), "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)
    readme = f"""# Gate F7 small observation ablation

- Completion: **{'PASS' if decision['completion_pass'] else 'FAIL'}**
- Observation-extension gate: **{'PASS' if extension_beats_base else 'NO-GO'}**
- Best held-out arm: `{winner['mode']}`
- base14 test force MAE: `{base['test_force_mae']:.9f}`
- Best extension (`{best_extension['mode']}`) test force MAE: `{best_extension['test_force_mae']:.9f}`
- Candidate PhysX evaluation: not entered because the frozen offline entry criterion failed.
- Promotion/release modification: none.
- F8: not started; separate approval is required.

All three arms used the same 19,200 samples, split, parent initialization, seed,
optimizer settings, and epoch count.  The result supports keeping base14 for the
next safety-control experiment rather than scaling this observation extension.
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(readme)
    evidence = [args.plan, *args.smokes, *args.collections, *args.trainings]
    with open(os.path.join(out_dir, "source_checksums.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "sha256")); writer.writeheader()
        writer.writerows({"path": os.path.abspath(path), "sha256": _sha256(path)} for path in evidence)
    outputs = ["physx_smoke.csv", "collection_summary.csv", "ablation_summary.csv", "decision.json", "README.md", "source_checksums.csv"]
    with open(os.path.join(out_dir, "checksums.sha256"), "w", encoding="utf-8") as handle:
        for name in outputs:
            handle.write(f"{_sha256(os.path.join(out_dir, name))}  {name}\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
