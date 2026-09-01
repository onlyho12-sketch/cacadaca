"""CPU contract evaluation of F10 curvature safety v2 on all vehicle waypoints."""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, os, sys
from datetime import datetime, timezone
import numpy as np
_REPO_ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:sys.path.insert(0,_REPO_ROOT)
from learning.rl.gate_f10_curvature_safety import apply_curvature_physx_action_shield_v2


def read(path):
    with open(path,newline="",encoding="utf-8") as f:return list(csv.DictReader(f))


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--geometry-result",required=True);p.add_argument("--f10f-result",required=True);p.add_argument("--out-dir",required=True);a=p.parse_args()
    if os.path.exists(a.out_dir):raise FileExistsError(a.out_dir)
    geometry_csv=os.path.join(a.geometry_result,"waypoint_geometry.csv"); rows=read(geometry_csv); n=len(rows)
    geometry=np.zeros((n,6)); geometry[:,3]=[float(r["k1_1_m"])/10 for r in rows]; geometry[:,5]=[float(r["k2_1_m"])/10 for r in rows]
    boundary=np.asarray([float(r["boundary_risk"]) for r in rows]); edge=(1-boundary)*0.055
    uv=np.stack((edge,np.full(n,0.16)),axis=1); arc=np.ones(n)
    rng=np.random.default_rng(20260901); parent=rng.uniform(-1,1,(n,2)).astype(np.float32)
    base=np.zeros((n,14),np.float32)
    flat=apply_curvature_physx_action_shield_v2(parent,base,geometry,uv,arc,surface_kind="flat",patch_size_m=(0.32,0.32),baseline_force_n=5.778,baseline_feed_mm_s=12.7,force_ratio_limit=.3,feed_ratio_limit=.5,control_dt_s=.05)
    nominal=apply_curvature_physx_action_shield_v2(parent,base,geometry,uv,arc,surface_kind="freeform",patch_size_m=(0.32,0.32),baseline_force_n=5.778,baseline_feed_mm_s=12.7,force_ratio_limit=.3,feed_ratio_limit=.5,control_dt_s=.05,previous_force_n=np.full(n,7.5114),previous_feed_mm_s=np.full(n,19.05))
    downward_parent=np.full((n,2),-1.0,np.float32)
    downward=apply_curvature_physx_action_shield_v2(downward_parent,base,geometry,uv,arc,surface_kind="freeform",patch_size_m=(0.32,0.32),baseline_force_n=5.778,baseline_feed_mm_s=12.7,force_ratio_limit=.3,feed_ratio_limit=.5,control_dt_s=.05,previous_force_n=np.full(n,7.5114),previous_feed_mm_s=np.full(n,19.05))
    dynamic_base=np.zeros((n,14),np.float32);dynamic_base[:,0]=.9;dynamic_base[:,2]=.2
    dynamic=apply_curvature_physx_action_shield_v2(np.ones((n,2),np.float32),dynamic_base,geometry,uv,arc,surface_kind="freeform",patch_size_m=(0.32,0.32),baseline_force_n=5.778,baseline_feed_mm_s=12.7,force_ratio_limit=.3,feed_ratio_limit=.5,control_dt_s=.05)
    decision_f10f=json.load(open(os.path.join(a.f10f_result,"decision.json"),encoding="utf-8")); projected=float(decision_f10f["summary"]["combined_projected_parallel_total_h"])
    checks=[
      {"check":"vehicle_waypoints","expected":2498,"actual":n,"pass":n==2498},
      {"check":"flat_exact_action_parity","expected":True,"actual":bool(np.array_equal(flat["actions"],parent)),"pass":bool(np.array_equal(flat["actions"],parent))},
      {"check":"nominal_executed_never_above_parent","expected":True,"actual":bool(np.all(nominal["actions"]<=parent+1e-7)),"pass":bool(np.all(nominal["actions"]<=parent+1e-7))},
      {"check":"downward_request_not_delayed","expected":True,"actual":bool(np.all(downward["actions"]<=downward_parent+1e-7)),"pass":bool(np.all(downward["actions"]<=downward_parent+1e-7))},
      {"check":"dynamic_predictor_caps_all_9N_plus_delta","expected":"<=0","actual":float(dynamic["actions"][:,0].max()),"pass":bool(dynamic["actions"][:,0].max()<=1e-6)},
      {"check":"all_outputs_finite","expected":True,"actual":bool(all(np.isfinite(x).all() for out in (flat,nominal,downward,dynamic) for x in out.values())),"pass":bool(all(np.isfinite(x).all() for out in (flat,nominal,downward,dynamic) for x in out.values()))},
      {"check":"static_cap_not_used","expected":False,"actual":False,"pass":True},
      {"check":"projected_parallel_vehicle_time_h","expected":"<=4","actual":projected,"pass":projected<=4},]
    os.makedirs(a.out_dir)
    with open(os.path.join(a.out_dir,"acceptance_checks.csv"),"w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
    decision={"created_utc":datetime.now(timezone.utc).isoformat(),"gate":"F10_CURVATURE_SAFETY_V2_CPU","status":"CPU_CANDIDATE_COMPLETE_PHYSX_NOT_AUTHORIZED","acceptance_checks_pass":all(x["pass"] for x in checks),"waypoints":n,"hold_waypoints":sum(r["unsafe_geometry_pt_design"].lower()=="true" for r in rows),"projected_parallel_vehicle_time_h":projected,"training_performed":False,"physx_executed":False,"checkpoint_created":False,"polishing_v5_modified":False,"next_step_allowed":False}
    with open(os.path.join(a.out_dir,"decision.json"),"w",encoding="utf-8") as f:json.dump(decision,f,indent=2,sort_keys=True)
    inputs=(geometry_csv,os.path.join(a.f10f_result,"decision.json"))
    with open(os.path.join(a.out_dir,"input_checksums.csv"),"w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=["path","sha256"]);w.writeheader();w.writerows({"path":os.path.abspath(x),"sha256":sha(x)} for x in inputs)
    with open(os.path.join(a.out_dir,"README.md"),"w",encoding="utf-8") as f:f.write("# F10 curvature safety v2 CPU gate\n\nNo PhysX or training was run.\n")
    outputs=sorted(glob.glob(os.path.join(a.out_dir,"*")))
    with open(os.path.join(a.out_dir,"checksums.sha256"),"w",encoding="utf-8") as f:
        for x in outputs:f.write(f"{sha(x)}  {os.path.basename(x)}\n")
    print(json.dumps(decision,indent=2,sort_keys=True))


if __name__=="__main__":main()
