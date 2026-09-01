"""F10-G2 v2 cylinder overload trace at the 120 Hz physics substep."""
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
parser.add_argument("--direction-mode", choices=("same_xx", "cross_xy"), default="same_xx")
parser.add_argument("--surface-seed-base", type=int, default=40000)
parser.add_argument("--physics-seed", type=int, default=20260901)
parser.add_argument("--policy-seed", type=int, default=20260831)
parser.add_argument("--max-control-steps", type=int, default=600)
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
    SafetyConfig, apply_curvature_physx_action_shield_v2)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(env, name, index, default=float("nan")):
    value = getattr(env, name, None)
    if value is None:
        return default
    try:
        return float(value[index])
    except Exception:
        return default


class TracedEnv(GateF7CurvedObservationEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        self.trace_rows = []
        self.trace_latch_events = []
        self.trace_active = set()
        self.trace_control_step = 0
        self.trace_substep = 0
        self.trace_values = {}
        self.trace_enabled = False
        super().__init__(cfg, render_mode, **kwargs)

    def _apply_action(self):
        before = {i: bool(self._force_hard_violated[i]) for i in self.trace_active}
        super()._apply_action()
        if not self.trace_enabled:
            return
        for i in sorted(self.trace_active):
            latched = bool(self._force_hard_violated[i])
            row = {
                "env": i,
                "control_step": self.trace_control_step,
                "substep": self.trace_substep,
                "force_sensor_raw_n": scalar(self, "_force_sensor_n", i),
                "force_sensor_filt_n": scalar(self, "_force_sensor_filt_n", i),
                "force_model_n": scalar(self, "_force_model_n", i),
                "force_used_n": scalar(self, "_force_used_n", i),
                "force_cmd_n": scalar(self, "_force_cmd", i),
                "pad_gap_m": scalar(self, "_pad_gap_m", i),
                "pad_in_patch": int(scalar(self, "_pad_in_patch", i, 0.0) > 0.5),
                "normal_alignment_error_deg": scalar(self, "_normal_alignment_error_deg", i),
                "sensor_fault": int(scalar(self, "_sensor_fault", i, 0.0) > 0.5),
                "hard_latched_before": int(before.get(i, False)),
                "hard_latched_after": int(latched),
                "latch_event": int(latched and not before.get(i, False)),
            }
            for name, values in self.trace_values.items():
                row[name] = float(values[i])
            self.trace_rows.append(row)
            if row["latch_event"]:
                self.trace_latch_events.append(dict(row))
        self.trace_substep += 1


def main():
    checkpoint = os.path.abspath(args.checkpoint)
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(out_dir)

    cfg = GateF7CurvedObservationEnvCfg()
    cfg.factory_profile_ids = PROFILE_IDS
    cfg.factory_envs_per_profile = 1
    cfg.scene.num_envs = len(PROFILE_IDS)
    cfg.factory_path_mode = "gate_c"
    cfg.factory_gate_c_direction_mode = args.direction_mode
    cfg.surface_kind = "cylinder"
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
    ckpt = torch.load(checkpoint, map_location=str(env.device), weights_only=False)
    state = ckpt["actor_state_dict"]
    dummy = TensorDict({"policy": torch.zeros(1, 14, device=env.device)}, batch_size=[1])
    actor = MLPModel(
        dummy, {"actor": ["policy"]}, "actor", 2, hidden_dims=[128, 128],
        activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3,
                          "std_type": "scalar"}).to(env.device)
    actor.load_state_dict(state, strict=True)
    actor.eval()

    torch.manual_seed(args.policy_seed)
    obs, _ = env.reset()
    profiles = [str(env._factory_metadata[i]["profile_id"]) for i in range(env.num_envs)]
    seeds = [int(env._factory_metadata[i]["profile_seed"]) for i in range(env.num_envs)]
    active = set(range(env.num_envs))
    env.trace_active = set(active)
    env.trace_enabled = True
    previous_force = None
    previous_feed = None
    safety_cfg = SafetyConfig()
    steps = 0
    try:
        for step in range(args.max_control_steps):
            if not active:
                break
            full = obs["policy"]
            with torch.no_grad():
                parent = actor(TensorDict({"policy": full}, batch_size=[env.num_envs])).clamp(-1, 1)
            parent_np = parent.detach().cpu().numpy()
            shield = apply_curvature_physx_action_shield_v2(
                parent_np, full.detach().cpu().numpy(),
                env.gate_f7_geometry().detach().cpu().numpy(),
                env._pad_uv_actual.detach().cpu().numpy(), env._arc.detach().cpu().numpy(),
                surface_kind="cylinder", patch_size_m=cfg.patch_size_m,
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
            env.trace_active = set(active)
            env.trace_values = {
                "parent_force_action": parent_np[:, 0],
                "executed_force_action": executed_np[:, 0],
                "executed_feed_action": executed_np[:, 1],
                "geometry_risk": shield["risk"],
                "force_cap_action": shield["force_cap_action"],
                "predictive_force_cap_action": shield["predictive_force_cap_action"],
                "predicted_force_n": shield["predicted_force_n"],
                "shield_force_command_n": shield["force_command_n"],
                "shield_feed_command_mm_s": shield["feed_command_mm_s"],
                "observed_force_mean_n": full[:, 0].detach().cpu().numpy() * 10.0,
                "observed_force_delta_n": full[:, 2].detach().cpu().numpy() * 5.0,
            }
            obs, _, terminated, truncated, _ = env.step(executed)
            steps = step + 1
            for i in (terminated | truncated).nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
                active.discard(i)
            if steps % 100 == 0:
                print(f"[F10G2 trace] step={steps} active={len(active)} rows={len(env.trace_rows)}", flush=True)

        rows = env.trace_rows
        if not rows:
            raise RuntimeError("no trace rows")
        for row in rows:
            row.update({"shield_mode": "curvature_safety_v2", "surface_kind": "cylinder",
                        "direction_mode": args.direction_mode,
                        "surface_profile": profiles[row["env"]],
                        "profile_seed": seeds[row["env"]]})
        os.makedirs(out_dir)
        with open(os.path.join(out_dir, "substep_trace.csv"), "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "gate": "F10G2_V2_CYLINDER_OVERLOAD_SUBSTEP_TRACE",
            "checkpoint": checkpoint, "checkpoint_sha256": sha256(checkpoint),
            "direction_mode": args.direction_mode, "surface_seed_base": args.surface_seed_base,
            "physics_seed": args.physics_seed, "policy_seed": args.policy_seed,
            "physics_dt": float(cfg.physical_sim_dt), "decimation": int(cfg.physical_decimation),
            "control_steps_executed": steps, "trace_rows": len(rows),
            "latch_events": env.trace_latch_events,
            "training_performed": False, "release_modified": False,
        }
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)
        print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
