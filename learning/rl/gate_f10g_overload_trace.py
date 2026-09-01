"""Gate F10-G: substep trace of the rejected curvature-safety freeform pilot.

The F9 evaluation samples at the 20 Hz control rate, but the hard-force latch in
``robot_polish_env`` fires on the 120 Hz physics substep.  Overload sequences
therefore recorded a sub-14 N ``raw_force_max_n`` while still latching a
violation.  This script re-runs a single (arm, geometry, direction, seed)
configuration with per-substep instrumentation so the spike itself is visible.

Read-only with respect to every protected artefact: it loads the frozen Gate E7
release checkpoint, reuses the frozen Gate F8 shield, and writes only into its
own new output directory.  No training, no promotion, no release change.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--surface-kind", choices=("freeform",), default="freeform")
parser.add_argument("--direction-mode", choices=("same_xx", "cross_xy"), required=True)
parser.add_argument("--surface-seed-base", type=int, required=True)
parser.add_argument("--physics-seed", type=int, default=20260901)
parser.add_argument("--policy-seed", type=int, default=20260831)
parser.add_argument("--max-control-steps", type=int, default=8000)
parser.add_argument("--out-dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.models.mlp_model import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from learning.polytwin.factory_surface_profiles import PROFILE_IDS  # noqa: E402
from learning.rl.env.gate_f7_curved_observation_env import GateF7CurvedObservationEnv  # noqa: E402
from learning.rl.env.gate_f7_curved_observation_env_cfg import GateF7CurvedObservationEnvCfg  # noqa: E402
from learning.rl.gate_f7_observation import BASE14, observation_dim  # noqa: E402
from learning.rl.gate_f10_curvature_safety import (  # noqa: E402
    SafetyConfig, apply_curvature_physx_action_shield)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _f(env, name: str, i: int, default: float = float("nan")) -> float:
    t = getattr(env, name, None)
    if t is None:
        return default
    try:
        return float(t[i])
    except Exception:
        return default


class TracedEnv(GateF7CurvedObservationEnv):
    """Records one row per physics substep per still-active environment."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.trace_rows: list[dict] = []
        self.trace_latch_events: list[dict] = []
        self.trace_active: set[int] = set()
        self.trace_control_step = 0
        self.trace_substep = 0
        self.trace_executed = None
        self.trace_parent = None
        self.trace_risk = None
        self.trace_force_cap = None
        self.trace_force_command = None
        self.trace_feed_command = None
        self.trace_enabled = False
        super().__init__(cfg, render_mode, **kwargs)

    def _apply_action(self) -> None:
        pre_latch = {i: bool(self._force_hard_violated[i]) for i in self.trace_active}
        super()._apply_action()
        if not self.trace_enabled:
            return
        for i in sorted(self.trace_active):
            latched_now = bool(self._force_hard_violated[i])
            row = {
                "env": i,
                "control_step": self.trace_control_step,
                "substep": self.trace_substep,
                "force_sensor_raw_n": _f(self, "_force_sensor_n", i),
                "force_sensor_filt_n": _f(self, "_force_sensor_filt_n", i),
                "force_model_n": _f(self, "_force_model_n", i),
                "force_used_n": _f(self, "_force_used_n", i),
                "force_cmd_n": _f(self, "_force_cmd", i),
                "pad_gap_m": _f(self, "_pad_gap_m", i),
                "pad_in_patch": int(_f(self, "_pad_in_patch", i, 0.0) > 0.5),
                "normal_alignment_error_deg": _f(self, "_normal_alignment_error_deg", i),
                "sensor_fault": int(_f(self, "_sensor_fault", i, 0.0) > 0.5),
                "parent_force_action": (float(self.trace_parent[i, 0])
                                        if self.trace_parent is not None else float("nan")),
                "executed_force_action": (float(self.trace_executed[i, 0])
                                          if self.trace_executed is not None else float("nan")),
                "executed_feed_action": (float(self.trace_executed[i, 1])
                                         if self.trace_executed is not None else float("nan")),
                "geometry_risk": (float(self.trace_risk[i])
                                  if self.trace_risk is not None else float("nan")),
                "force_cap_action": (float(self.trace_force_cap[i])
                                     if self.trace_force_cap is not None else float("nan")),
                "shield_force_command_n": (float(self.trace_force_command[i])
                                           if self.trace_force_command is not None else float("nan")),
                "shield_feed_command_mm_s": (float(self.trace_feed_command[i])
                                             if self.trace_feed_command is not None else float("nan")),
                "hard_latched_before": int(pre_latch.get(i, False)),
                "hard_latched_after": int(latched_now),
                "latch_event": int(latched_now and not pre_latch.get(i, False)),
            }
            self.trace_rows.append(row)
            if row["latch_event"]:
                self.trace_latch_events.append(dict(row))
        self.trace_substep += 1


def main() -> None:
    ckpt = os.path.abspath(args.checkpoint)
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)

    cfg = GateF7CurvedObservationEnvCfg()
    cfg.factory_profile_ids = PROFILE_IDS
    cfg.factory_envs_per_profile = 1
    cfg.scene.num_envs = len(PROFILE_IDS)
    cfg.factory_path_mode = "gate_c"
    cfg.factory_gate_c_direction_mode = args.direction_mode
    cfg.surface_kind = args.surface_kind
    cfg.align_pad_to_surface_normal = True
    cfg.gate_f7_observation_mode = BASE14
    cfg.observation_space = observation_dim(BASE14)
    cfg.surface_seed_base = args.surface_seed_base
    cfg.seed = args.physics_seed
    cfg.enable_pad_physical_contact = True
    cfg.repolish_max_passes = 1
    cfg.episode_length_s = 840.0

    env = TracedEnv(cfg, render_mode=None)
    env._repolish_mode = True

    ck = torch.load(ckpt, map_location=str(env.device), weights_only=False)
    state = ck["actor_state_dict"]
    if tuple(state["mlp.0.weight"].shape) != (128, 14):
        raise ValueError("parent must be a 14-D actor")
    dummy = TensorDict({"policy": torch.zeros(1, 14, device=env.device)}, batch_size=[1])
    actor = MLPModel(dummy, {"actor": ["policy"]}, "actor", 2,
                     hidden_dims=[128, 128], activation="elu", obs_normalization=True,
                     distribution_cfg={"class_name": "GaussianDistribution",
                                       "init_std": 0.3, "std_type": "scalar"}).to(env.device)
    actor.load_state_dict(state, strict=True)
    actor.eval()

    torch.manual_seed(args.policy_seed)
    obs, _ = env.reset()
    seeds = [int(env._factory_metadata[i]["profile_seed"]) for i in range(env.num_envs)]
    profiles = [str(env._factory_metadata[i]["profile_id"]) for i in range(env.num_envs)]

    active = set(range(env.num_envs))
    env.trace_active = set(active)
    env.trace_enabled = True
    safety_cfg = SafetyConfig()
    previous_force = None
    previous_feed = None
    control_steps_executed = 0
    try:
        for step in range(args.max_control_steps):
            if not active:
                break
            full = obs["policy"]
            with torch.no_grad():
                parent = actor(TensorDict({"policy": full},
                                          batch_size=[env.num_envs])).clamp(-1.0, 1.0)
            parent_np = parent.detach().cpu().numpy()
            shield = apply_curvature_physx_action_shield(
                parent_np, env.gate_f7_geometry().detach().cpu().numpy(),
                env._pad_uv_actual.detach().cpu().numpy(),
                env._arc.detach().cpu().numpy(), surface_kind=args.surface_kind,
                patch_size_m=cfg.patch_size_m,
                baseline_force_n=float(env.recipe.target_contact_force_n),
                baseline_feed_mm_s=float(env.recipe.feed_speed_mm_s),
                force_ratio_limit=float(cfg.force_ratio_limit),
                feed_ratio_limit=float(cfg.feed_ratio_limit),
                control_dt_s=float(env.quality_dt), previous_force_n=previous_force,
                previous_feed_mm_s=previous_feed, cfg=safety_cfg)
            executed_np = shield["actions"]
            previous_force = np.asarray(shield["force_command_n"]).copy()
            previous_feed = np.asarray(shield["feed_command_mm_s"]).copy()
            executed = torch.as_tensor(executed_np, device=env.device)
            for i in sorted(set(range(env.num_envs)) - active):
                executed[i] = 0.0
            env.trace_control_step = step
            env.trace_substep = 0
            env.trace_parent = parent_np
            env.trace_executed = executed_np
            env.trace_risk = shield["risk"]
            env.trace_force_cap = shield["force_cap_action"]
            env.trace_force_command = shield["force_command_n"]
            env.trace_feed_command = shield["feed_command_mm_s"]
            env.trace_active = set(active)
            obs, _, terminated, truncated, _ = env.step(executed)
            control_steps_executed = step + 1
            for i in (terminated | truncated).nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
                active.discard(i)
            if (step + 1) % 200 == 0:
                print(f"[F10G trace] curvature_safety/{args.surface_kind}/{args.direction_mode}/"
                      f"seed{args.surface_seed_base} step={step+1} active={len(active)} "
                      f"rows={len(env.trace_rows)}", flush=True)
        rows = env.trace_rows
        if not rows:
            raise RuntimeError("no trace rows recorded")
        for r in rows:
            r["shield_mode"] = "curvature_safety"
            r["surface_kind"] = args.surface_kind
            r["direction_mode"] = args.direction_mode
            r["surface_seed_base"] = args.surface_seed_base
            r["surface_profile"] = profiles[r["env"]]
            r["profile_seed"] = seeds[r["env"]]
        fields = list(rows[0].keys())
        with open(os.path.join(out_dir, "substep_trace.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        # The base environment may auto-reset a finished env during env.step().
        # Derive metadata from events captured inside _apply_action, before reset.
        latch_step = {}
        for event in env.trace_latch_events:
            latch_step.setdefault(int(event["env"]), int(event["control_step"]))
        meta = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "gate": "F10G_OVERLOAD_SUBSTEP_TRACE",
            "shield_mode": "curvature_safety", "surface_kind": args.surface_kind,
            "direction_mode": args.direction_mode,
            "surface_seed_base": args.surface_seed_base,
            "physics_seed": args.physics_seed, "policy_seed": args.policy_seed,
            "checkpoint": ckpt, "checkpoint_sha256": _sha256(ckpt),
            "observation_mode": BASE14, "profiles": profiles, "profile_seeds": seeds,
            "physics_dt": float(cfg.physical_sim_dt),
            "decimation": int(cfg.physical_decimation),
            "trace_rows": len(rows),
            "max_control_steps_requested": int(args.max_control_steps),
            "control_steps_executed": int(control_steps_executed),
            "latch_event_count": len(env.trace_latch_events),
            "latch_control_step_by_env": {str(k): int(v) for k, v in latch_step.items()},
            "latched_envs": sorted(int(k) for k in latch_step),
            "training_performed": False, "champion_promoted": False,
            "release_modified": False,
        }
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
        print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
