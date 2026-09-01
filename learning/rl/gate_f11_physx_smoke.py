"""Small PhysX-to-F11 integration smoke using deterministic G3 only.

The established environment is subclassed without modification.  Its final ROI
SurfaceState is copied before automatic reset, then passed through the standalone
F11 full-cell inspection and bounded failed-region planner.  This smoke does not
execute a second physical pass and does not integrate polishing_v5.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone


parser = argparse.ArgumentParser()
parser.add_argument("--parent-checkpoint", required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--profile", default="factory_prepolish")
parser.add_argument("--surface-seed-base", type=int, default=71000)
parser.add_argument("--physics-seed", type=int, default=20261001)
parser.add_argument("--max-control-steps", type=int, default=8000)
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

# Import the already smoke-validated F10-H evaluation environment while letting
# it own the single Isaac Lab AppLauncher instance.
saved_argv = sys.argv
sys.argv = [
    saved_argv[0], "--parent-checkpoint", args.parent_checkpoint,
    "--arm", "deterministic_g3", "--surface-kind", "cylinder",
    "--direction-mode", "same_xx", "--out-dir", args.out_dir,
]
if args.headless:
    sys.argv.append("--headless")
import learning.rl.gate_f10h_residual_eval as f10h  # noqa: E402
sys.argv = saved_argv

import numpy as np  # noqa: E402
import torch  # noqa: E402

from learning.rl.env.planar_roi_diagnostics import surface_view  # noqa: E402
from learning.rl.gate_f11_selective_pipeline import (  # noqa: E402
    CellGeometry,
    begin_pipeline,
    ui_cell_rows,
)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class F11CaptureEnv(f10h.ResidualEvalEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        self.final_surfaces = {}
        super().__init__(cfg, render_mode, **kwargs)

    def capture_final(self, env_id):
        self.final_surfaces[env_id] = surface_view(
            self._surfaces[env_id], self._roi_slices).copy()
        super().capture_final(env_id)


def cylinder_geometry(state, radius_m=0.60):
    geometry = CellGeometry.from_surface_state(state)
    shape = state.shape
    # Sign follows the current F10 adapter convention: outward convex is negative.
    return CellGeometry(
        normal_xyz=geometry.normal_xyz,
        k1_1_m=np.zeros(shape, dtype=np.float64),
        k2_1_m=np.full(shape, -1.0 / radius_m, dtype=np.float64),
        curvature_radius_m=np.full(shape, radius_m, dtype=np.float64),
        boundary_risk=np.zeros(shape, dtype=np.float64),
        geometry_risk=np.full(shape, 0.055 / radius_m, dtype=np.float64),
        fit_confidence=np.ones(shape, dtype=np.float64),
        surface_class=np.full(shape, "convex", dtype="U32"),
    )


def main():
    parent = os.path.abspath(args.parent_checkpoint)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isfile(parent):
        raise FileNotFoundError(parent)
    if os.path.exists(out_dir):
        raise FileExistsError(out_dir)

    cfg = f10h.smoke.GateF7CurvedObservationEnvCfg()
    cfg.factory_profile_ids = (args.profile,)
    cfg.factory_envs_per_profile = 1
    cfg.scene.num_envs = 1
    cfg.factory_path_mode = "gate_c"
    cfg.factory_gate_c_direction_mode = "same_xx"
    cfg.surface_kind = "cylinder"
    cfg.curvature_radius_m = 0.60
    cfg.align_pad_to_surface_normal = True
    cfg.gate_f7_observation_mode = f10h.smoke.NORMAL_CURVATURE20
    cfg.observation_space = f10h.smoke.observation_dim(f10h.smoke.NORMAL_CURVATURE20)
    cfg.surface_seed_base = args.surface_seed_base
    cfg.seed = args.physics_seed
    cfg.enable_pad_physical_contact = True
    cfg.repolish_max_passes = 1
    cfg.episode_length_s = 840.0

    F11CaptureEnv.parent_checkpoint = parent
    F11CaptureEnv.control_parent = False
    env = F11CaptureEnv(cfg, render_mode=None)
    env._repolish_mode = True
    completed = False
    steps = 0
    try:
        obs, _ = env.reset()
        for step in range(args.max_control_steps):
            policy = obs["policy"]
            if policy.shape != (1, 20) or not bool(torch.isfinite(policy).all()):
                raise RuntimeError("invalid normal_curvature20 observation")
            actions = torch.zeros(1, 2, device=env.device)
            obs, _, terminated, truncated, _ = env.step(actions)
            steps = step + 1
            if bool((terminated | truncated)[0]):
                completed = True
                break
            if steps % 1000 == 0:
                print(f"[F11 PhysX smoke] step={steps}", flush=True)

        if not completed or 0 not in env.final_surfaces:
            raise RuntimeError(f"first pass incomplete after {steps} control steps")
        final_state = env.final_surfaces[0]
        reached = np.asarray(final_state.dwell_time_s) > 0.0
        pipeline = begin_pipeline(
            final_state,
            geometry=cylinder_geometry(final_state),
            reached_mask=reached,
            surface_id=f"{args.profile}:cylinder:same_xx:{args.surface_seed_base}",
        )
        os.makedirs(out_dir)
        write_csv(os.path.join(out_dir, "cell_quality_ui.csv"), ui_cell_rows(pipeline))
        write_csv(os.path.join(out_dir, "rework_regions.csv"), pipeline.regions)
        write_csv(os.path.join(out_dir, "rework_path.csv"), pipeline.rework_path)
        state_total = sum(int(v) for v in {
            name: int((pipeline.ledger.status == name).sum())
            for name in np.unique(pipeline.ledger.status)
        }.values())
        status_counts = {
            str(name): int((pipeline.ledger.status == name).sum())
            for name in np.unique(pipeline.ledger.status)
        }
        all4_pass_cells = int(pipeline.ledger.quality["all4_pass"].sum())
        safety = env.latched_safety.get(0, {})
        result = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "gate": "F11_PHYSX_TO_SELECTIVE_PLANNER_SMOKE",
            "status": "PASS" if completed else "FAIL",
            "scope": "FIRST_PASS_PHYSX_TO_F11_PLAN_ONLY",
            "second_pass_physx_executed": False,
            "polishing_v5_modified": False,
            "profile": args.profile,
            "surface_kind": "cylinder",
            "direction_mode": "same_xx",
            "surface_seed_base": args.surface_seed_base,
            "physics_seed": args.physics_seed,
            "control_steps": steps,
            "cells": int(final_state.micro_height_um.size),
            "reached_cells": int(reached.sum()),
            "unreached_cells": int((~reached).sum()),
            "ledger_cells": state_total,
            "ledger_complete": state_total == final_state.micro_height_um.size,
            "ledger_status_counts": status_counts,
            "all4_pass_cells": all4_pass_cells,
            "all4_pass_area_pct": 100.0 * all4_pass_cells / state_total,
            "planner_summary": pipeline.planner_summary,
            "f10h_cap_contract_failures": env._f10h_cap_contract_failures,
            "f10h_supervisor_faults": env._f10h_supervisor_faults,
            "f10h_force_overload_seen": env._f10h_force_overload_seen,
            "latched_safety": safety,
            "parent_checkpoint": parent,
            "parent_checkpoint_sha256": f10h.smoke.sha256(parent),
            "actual_gu_sensor_connected": False,
        }
        result["smoke_pass"] = bool(
            result["ledger_complete"]
            and result["f10h_cap_contract_failures"] == 0
            and result["f10h_supervisor_faults"] == 0
            and result["f10h_force_overload_seen"] == 0
            and not any(bool(v) for v in safety.values())
        )
        if not result["smoke_pass"]:
            result["status"] = "FAIL"
        elif pipeline.planner_summary["automatic_rework_allowed"]:
            result["status"] = "ADAPTER_PASS_REWORK_READY"
        else:
            result["status"] = "ADAPTER_PASS_REWORK_BLOCKED"
        with open(os.path.join(out_dir, "decision.json"), "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        if not result["smoke_pass"]:
            raise RuntimeError("F11 PhysX smoke acceptance failed")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        f10h.smoke.app.close()
