"""Replay the F10-G2 trace through the Isaac-free F10-G3 supervisor."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from learning.rl.gate_f10_substep_supervisor import (
    SubstepSafetyState,
    supervise_substep,
)


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if os.path.exists(args.out_dir):
        raise FileExistsError(args.out_dir)
    with open(args.trace, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    summaries = []
    contract_ok = True
    interception_ok = True
    for env in sorted({int(row["env"]) for row in rows}):
        selected = [row for row in rows if int(row["env"]) == env]
        state = SubstepSafetyState.zeros(1)
        intervention_indices = []
        transition_indices = []
        previous_intervention = False
        latch_indices = []
        for index, row in enumerate(selected):
            parent = float(row["shield_force_command_n"])
            result = supervise_substep(
                [parent], [float(row["force_sensor_raw_n"])],
                [float(row["force_sensor_filt_n"])], [float(row["pad_gap_m"])],
                surface_kind="cylinder", state=state)
            active = bool(result["intervention"][0])
            if active:
                intervention_indices.append(index)
            if active and not previous_intervention:
                transition_indices.append(index)
            previous_intervention = active
            contract_ok &= float(result["force_command_n"][0]) <= parent + 1e-12
            if row["latch_event"] == "1":
                latch_indices.append(index)
        lead = None
        if latch_indices:
            latch = latch_indices[0]
            prior = [index for index in transition_indices if index < latch]
            lead = latch - prior[-1] if prior else -1
            interception_ok &= lead > 0
        summaries.append({
            "env": env,
            "surface_profile": selected[0]["surface_profile"],
            "trace_rows": len(selected),
            "original_latch_events": len(latch_indices),
            "supervisor_intervention_rows": len(intervention_indices),
            "supervisor_intervention_fraction": len(intervention_indices) / len(selected),
            "supervisor_lead_substeps_before_latch": "" if lead is None else lead,
        })
    max_fraction = max(row["supervisor_intervention_fraction"] for row in summaries)
    checks = {
        "trace_rows_14058": len(rows) == 14058,
        "single_original_latch": sum(int(row["original_latch_events"]) for row in summaries) == 1,
        "supervisor_intercepts_before_latch": interception_ok,
        "force_command_never_above_parent": contract_ok,
        "intervention_fraction_each_env_le_2pct": max_fraction <= 0.02,
    }
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F10G3_SUBSTEP_SUPERVISOR_OFFLINE_REPLAY",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "max_intervention_fraction": max_fraction,
        "trace_sha256": sha256(args.trace),
        "training_performed": False,
        "physx_performed": False,
        "production_ready": False,
        "next_required": "isolated PhysX adapter and frozen small revalidation",
    }
    os.makedirs(args.out_dir)
    with open(os.path.join(args.out_dir, "replay_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with open(os.path.join(args.out_dir, "decision.json"), "w", encoding="utf-8") as fh:
        json.dump(decision, fh, indent=2, sort_keys=True)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
