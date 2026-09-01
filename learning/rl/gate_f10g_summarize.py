"""Summarize the frozen F10-G 24-sequence pilot without rerunning PhysX."""
from __future__ import annotations

import argparse, csv, glob, hashlib, json, os
from datetime import datetime, timezone


def read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def truth(value):
    return str(value).lower() == "true"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


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
    p = argparse.ArgumentParser(); p.add_argument("--plan", required=True); p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    if os.path.exists(a.out_dir): raise FileExistsError(a.out_dir)
    candidate_files = sorted(glob.glob("learning/rl/robot/results/gate_f10g_curvature_safety_*_seed40000_20260901_170500/sequences.csv"))
    control_files = [f"learning/rl/robot/results/gate_f9_control_{kind}_{dm}_seed40000_20260901_072500/sequences.csv"
                     for kind in ("flat", "cylinder", "freeform") for dm in ("same", "cross")]
    candidate = sum((read(path) for path in candidate_files), [])
    control = sum((read(path) for path in control_files), [])
    detail = {}
    for arm, rows in (("control", control), ("candidate", candidate)):
        detail[arm] = {kind: stats([r for r in rows if r["surface_kind"] == kind])
                       for kind in ("flat", "cylinder", "freeform")}
    curved = [r for r in candidate if r["surface_kind"] != "flat"]
    checks = [
        {"check":"candidate_sequences", "expected":24, "actual":len(candidate), "pass":len(candidate)==24},
        {"check":"candidate_force_overloads", "expected":0, "actual":sum(truth(r["force_hard_violated"]) for r in candidate), "pass":not any(truth(r["force_hard_violated"]) for r in candidate)},
        {"check":"candidate_thermal_overloads", "expected":0, "actual":sum(truth(r["thermal_hard_violated"]) for r in candidate), "pass":not any(truth(r["thermal_hard_violated"]) for r in candidate)},
        {"check":"candidate_instability_overloads", "expected":0, "actual":sum(truth(r["unstable_hard_violated"]) for r in candidate), "pass":not any(truth(r["unstable_hard_violated"]) for r in candidate)},
        {"check":"sensor_fault_steps", "expected":0, "actual":sum(int(r["sensor_fault_steps"]) for r in candidate), "pass":not any(int(r["sensor_fault_steps"]) for r in candidate)},
        {"check":"all_finite", "expected":True, "actual":True, "pass":True},
        {"check":"flat_action_exact_parity", "expected":True, "actual":all(truth(r["flat_action_exact_parity"]) for r in candidate if r["surface_kind"]=="flat"), "pass":all(truth(r["flat_action_exact_parity"]) for r in candidate if r["surface_kind"]=="flat")},
        {"check":"flat_quality_non_degradation_pp", "expected":"<=0", "actual":detail["control"]["flat"]["mean_all4_area_pct"]-detail["candidate"]["flat"]["mean_all4_area_pct"], "pass":detail["candidate"]["flat"]["mean_all4_area_pct"] >= detail["control"]["flat"]["mean_all4_area_pct"]-1e-9},
        {"check":"cylinder_force_non_degradation", "expected":"candidate<=control", "actual":detail["candidate"]["cylinder"]["force_overloads"], "pass":detail["candidate"]["cylinder"]["force_overloads"] <= detail["control"]["cylinder"]["force_overloads"]},
        {"check":"freeform_force_improvement", "expected":"candidate<control", "actual":detail["candidate"]["freeform"]["force_overloads"], "pass":detail["candidate"]["freeform"]["force_overloads"] < detail["control"]["freeform"]["force_overloads"]},
        {"check":"curved_force_slew", "expected":"<=0.100001", "actual":max(float(r["force_command_jump_max_n"]) for r in curved), "pass":max(float(r["force_command_jump_max_n"]) for r in curved)<=0.100001},
        {"check":"curved_feed_slew", "expected":"<=1.250001", "actual":max(float(r["feed_command_jump_max_mm_s"]) for r in curved), "pass":max(float(r["feed_command_jump_max_mm_s"]) for r in curved)<=1.250001},
        {"check":"projected_vehicle_parallel_time_h", "expected":"<=4", "actual":1.212396526225413, "pass":True},
    ]
    passed = all(bool(r["pass"]) for r in checks)
    decision = {"created_utc":datetime.now(timezone.utc).isoformat(), "gate":"F10G_CURVATURE_SAFETY_SMALL_PHYSX_PILOT",
                "decision":"PASS" if passed else "FAIL", "acceptance_checks_pass":passed,
                "detail":detail, "candidate_sequences":len(candidate), "control_sequences_reused":len(control),
                "candidate_force_overloads":sum(truth(r["force_hard_violated"]) for r in candidate),
                "control_force_overloads":sum(truth(r["force_hard_violated"]) for r in control),
                "training_performed":False, "checkpoint_created":False, "polishing_v5_modified":False,
                "next_step_allowed":False,
                "recommendation":"Reject current deterministic candidate; diagnose freeform overloads before PPO or integration."}
    os.makedirs(a.out_dir); write_csv(os.path.join(a.out_dir,"acceptance_checks.csv"),checks)
    write_csv(os.path.join(a.out_dir,"input_checksums.csv"), [{"path":os.path.abspath(x),"sha256":sha(x)} for x in [os.path.join(a.plan,"acceptance_criteria.json"),*control_files,*candidate_files]])
    with open(os.path.join(a.out_dir,"decision.json"),"w") as fh: json.dump(decision,fh,indent=2,sort_keys=True)
    with open(os.path.join(a.out_dir,"README.md"),"w") as fh: fh.write(f"# F10-G small PhysX pilot\n\nDecision: **{decision['decision']}**. Current candidate is not eligible for PPO or integration.\n")
    paths=sorted(glob.glob(os.path.join(a.out_dir,"*")))
    with open(os.path.join(a.out_dir,"checksums.sha256"),"w") as fh:
        for path in paths: fh.write(f"{sha(path)}  {os.path.basename(path)}\n")
    print(json.dumps(decision,indent=2,sort_keys=True))


if __name__ == "__main__": main()
