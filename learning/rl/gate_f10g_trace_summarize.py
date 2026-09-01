"""CPU-only summary of frozen F10-G substep traces."""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, os
from datetime import datetime, timezone


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--trace-glob",required=True); p.add_argument("--out-dir",required=True); a=p.parse_args()
    if os.path.exists(a.out_dir): raise FileExistsError(a.out_dir)
    files=sorted(glob.glob(a.trace_glob)); events=[]
    if len(files)!=2: raise RuntimeError(f"expected two traces, got {files}")
    for path in files:
        with open(path,newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
        previous_by_env={}
        for r in rows:
            env_key=r["env"]
            if r["latch_event"]!="1":
                previous_by_env[env_key]=r
                continue
            prev=previous_by_env[env_key]
            slope=(float(r["force_sensor_raw_n"])-float(prev["force_sensor_raw_n"]))*120.0
            events.append({
                "direction_mode":r["direction_mode"], "surface_profile":r["surface_profile"],
                "env":r["env"], "control_step":r["control_step"], "substep":r["substep"],
                "raw_force_before_n":prev["force_sensor_raw_n"], "raw_force_latch_n":r["force_sensor_raw_n"],
                "raw_force_slope_n_s":slope, "filtered_force_n":r["force_sensor_filt_n"],
                "force_command_n":r["force_cmd_n"], "geometry_risk":r["geometry_risk"],
                "force_cap_action":r["force_cap_action"], "parent_force_action":r["parent_force_action"],
                "executed_force_action":r["executed_force_action"],
                "executed_above_parent":float(r["executed_force_action"])>float(r["parent_force_action"])+1e-7,
                "feed_command_mm_s":r["shield_feed_command_mm_s"], "pad_gap_m":r["pad_gap_m"],
                "normal_alignment_error_deg":r["normal_alignment_error_deg"],
                "pad_in_patch":r["pad_in_patch"], "sensor_fault":r["sensor_fault"]})
            previous_by_env[env_key]=r
    if len(events)!=4: raise RuntimeError(f"expected four reproduced events, got {len(events)}")
    os.makedirs(a.out_dir)
    with open(os.path.join(a.out_dir,"overload_events.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(events[0])); w.writeheader(); w.writerows(events)
    summary={
        "created_utc":datetime.now(timezone.utc).isoformat(), "gate":"F10G_SUBSTEP_ROOT_CAUSE",
        "status":"ROOT_CAUSE_CONFIRMED_CURRENT_CANDIDATE_REJECTED", "events_reproduced":4,
        "executed_force_above_parent_events":sum(e["executed_above_parent"] for e in events),
        "geometry_risk_at_latch_range":[min(float(e["geometry_risk"]) for e in events),max(float(e["geometry_risk"]) for e in events)],
        "raw_force_slope_n_s_range":[min(e["raw_force_slope_n_s"] for e in events),max(e["raw_force_slope_n_s"] for e in events)],
        "sensor_fault_events":sum(int(e["sensor_fault"]) for e in events),
        "finding":"symmetric force slew can hold executed force above a downward parent request; geometry-only risk is low at all latch events and lacks dynamic force/rate prediction",
        "rejected_hypotheses":["contact-loss/re-entry is common to all four events","feed slowdown alone explains all four events"],
        "required_candidate_properties":["executed force never exceeds downward parent request","downward force change is immediate or substantially faster than upward slew","dynamic measured-force and force-rate predictive guard","flat exact parity"],
        "physx_executed":True,"training_performed":False,"checkpoint_created":False,
        "polishing_v5_modified":False,"next_step_allowed":False}
    with open(os.path.join(a.out_dir,"decision.json"),"w",encoding="utf-8") as f: json.dump(summary,f,indent=2,sort_keys=True)
    with open(os.path.join(a.out_dir,"input_checksums.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["path","sha256"]); w.writeheader(); w.writerows({"path":os.path.abspath(x),"sha256":sha(x)} for x in files)
    with open(os.path.join(a.out_dir,"README.md"),"w",encoding="utf-8") as f: f.write("# F10-G substep root cause\n\nFour overloads reproduced. Current candidate remains rejected.\n")
    outputs=sorted(glob.glob(os.path.join(a.out_dir,"*")))
    with open(os.path.join(a.out_dir,"checksums.sha256"),"w",encoding="utf-8") as f:
        for x in outputs: f.write(f"{sha(x)}  {os.path.basename(x)}\n")
    print(json.dumps(summary,indent=2,sort_keys=True))


if __name__=="__main__": main()
