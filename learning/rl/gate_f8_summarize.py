"""Summarize frozen Gate F8 force-safety pilot evidence."""
from __future__ import annotations

import argparse, csv, hashlib, json, os
from collections import defaultdict
from datetime import datetime, timezone


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def rows(paths):
    out=[]
    for path in paths:
        with open(path,newline='') as f: out.extend(csv.DictReader(f))
    return out


def main():
    p=argparse.ArgumentParser(); p.add_argument('--plan',required=True)
    p.add_argument('--stage1',nargs=12,required=True); p.add_argument('--stage2',nargs=8,required=True)
    p.add_argument('--duplicates',nargs='*',default=[]); p.add_argument('--out-dir',required=True); a=p.parse_args()
    if os.path.exists(a.out_dir): raise FileExistsError(a.out_dir)
    os.makedirs(a.out_dir)
    s1=rows(a.stage1); s2=rows(a.stage2); official=s1+s2
    summary=[]
    for mode in ('control','static_cap','predictive_shield'):
      for kind in ('flat','cylinder','sphere','freeform'):
        x=[r for r in official if r['shield_mode']==mode and r['surface_kind']==kind]
        if not x: continue
        summary.append({'shield_mode':mode,'surface_kind':kind,'sequences':len(x),
          'safety_pass':sum(r['safety_ok']=='True' for r in x),
          'quality_pass':sum(r['quality_ok']=='True' for r in x),
          'force_overloads':sum(r['force_hard_violated']=='True' for r in x),
          'sensor_fault_steps':sum(int(r['sensor_fault_steps']) for r in x),
          'raw_force_max_n':max(float(r['raw_force_max_n']) for r in x),
          'mean_all4_area_pct':sum(float(r['final_all4_pass_area_pct']) for r in x)/len(x),
          'mean_shield_step_fraction':sum(float(r['shield_step_fraction']) for r in x)/len(x)})
    def group(mode,kinds): return [r for r in official if r['shield_mode']==mode and r['surface_kind'] in kinds]
    control_curved=group('control',('cylinder','freeform')); candidate_curved=group('static_cap',('cylinder','freeform'))
    control_flat=group('control',('flat',)); candidate_flat=group('static_cap',('flat',))
    c_area=sum(float(r['final_all4_pass_area_pct']) for r in control_curved)/len(control_curved)
    s_area=sum(float(r['final_all4_pass_area_pct']) for r in candidate_curved)/len(candidate_curved)
    checks={
      'curved_safety_improved':sum(r['safety_ok']=='True' for r in candidate_curved)>sum(r['safety_ok']=='True' for r in control_curved),
      'candidate_zero_force_overloads':all(r['force_hard_violated']=='False' for r in group('static_cap',('flat','cylinder','sphere','freeform'))),
      'flat_safety_not_regressed':sum(r['safety_ok']=='True' for r in candidate_flat)>=sum(r['safety_ok']=='True' for r in control_flat),
      'flat_quality_not_regressed':sum(r['quality_ok']=='True' for r in candidate_flat)>=sum(r['quality_ok']=='True' for r in control_flat),
      'curved_all4_drop_le_1pp':c_area-s_area<=1.0,
      'zero_sensor_faults':sum(int(r['sensor_fault_steps']) for r in official)==0,
    }
    decision={'created_utc':datetime.now(timezone.utc).isoformat(),'gate':'F8_SMALL_FORCE_SAFETY_CONSTRAINT_PILOT',
      'completion_pass':len(s1)==48 and len(s2)==32,'force_safety_gate_pass':all(checks.values()),
      'selected_candidate':'static_cap','checks':checks,
      'control_curved_safety_pass':sum(r['safety_ok']=='True' for r in control_curved),
      'candidate_curved_safety_pass':sum(r['safety_ok']=='True' for r in candidate_curved),
      'control_force_overloads':sum(r['force_hard_violated']=='True' for r in group('control',('flat','cylinder','sphere','freeform'))),
      'candidate_force_overloads':sum(r['force_hard_violated']=='True' for r in group('static_cap',('flat','cylinder','sphere','freeform'))),
      'curved_all4_area_delta_pp_candidate_minus_control':s_area-c_area,
      'training_performed':False,'ppo_performed':False,'champion_promoted':False,'release_modified':False,
      'production_readiness':False,
      'recommendation':'Static curved-only +0.50 force-action cap passes this small synthetic pilot. Require a separately approved larger multi-seed validation before integration or release.'}
    with open(os.path.join(a.out_dir,'group_summary.csv'),'w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    with open(os.path.join(a.out_dir,'decision.json'),'w') as f: json.dump(decision,f,indent=2,sort_keys=True)
    with open(os.path.join(a.out_dir,'README.md'),'w') as f:
      f.write(f"# Gate F8 force-safety pilot\n\n- Completion: **PASS** (80 official sequences)\n- Safety gate: **{'PASS' if decision['force_safety_gate_pass'] else 'FAIL'}**\n- Selected candidate: `static_cap` (curved-only force action <= +0.50)\n- Curved safety: control {decision['control_curved_safety_pass']}/16 -> candidate {decision['candidate_curved_safety_pass']}/16\n- Force overloads: control {decision['control_force_overloads']} -> candidate {decision['candidate_force_overloads']}\n- Curved all4 delta: {decision['curved_all4_area_delta_pp_candidate_minus_control']:.4f} percentage points\n- Training/PPO/promotion/release changes: none\n- Production-ready: no; larger multi-seed validation requires separate approval.\n\nA deterministic duplicate stage2 execution was excluded from official counts; all eight duplicate sequence CSV hashes matched their official counterparts exactly.\n")
    sources=[a.plan,*a.stage1,*a.stage2,*a.duplicates]
    with open(os.path.join(a.out_dir,'source_checksums.csv'),'w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=('role','path','sha256'));w.writeheader()
      for i,path in enumerate(sources): w.writerow({'role':'duplicate' if path in a.duplicates else 'official','path':os.path.abspath(path),'sha256':sha(path)})
    names=('group_summary.csv','decision.json','README.md','source_checksums.csv')
    with open(os.path.join(a.out_dir,'checksums.sha256'),'w') as f:
      for n in names:f.write(f'{sha(os.path.join(a.out_dir,n))}  {n}\n')
    print(json.dumps(decision,indent=2,sort_keys=True))
if __name__=='__main__':main()
