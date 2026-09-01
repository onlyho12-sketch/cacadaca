"""Fixed-seed PhysX evaluation for deterministic G3 and F10-H residual PPO."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone


parser = argparse.ArgumentParser()
parser.add_argument("--parent-checkpoint", required=True)
parser.add_argument("--residual-checkpoint")
parser.add_argument("--arm", choices=("control_parent", "deterministic_g3", "residual_ppo"),
                    required=True)
parser.add_argument("--surface-kind", choices=("cylinder", "freeform"), required=True)
parser.add_argument("--direction-mode", choices=("same_xx", "cross_xy"), required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--envs-per-profile", type=int, default=1)
parser.add_argument("--surface-seed-base", type=int, default=60000)
parser.add_argument("--physics-seed", type=int, default=20260921)
parser.add_argument("--policy-seed", type=int, default=20260922)
parser.add_argument("--max-control-steps", type=int, default=8000)
parser.add_argument("--smoke-only", action="store_true")
parser.add_argument("--headless", action="store_true")
eval_args = parser.parse_args()

# Reuse the smoke-validated environment without starting a second AppLauncher.
saved_argv = sys.argv
sys.argv = [saved_argv[0], "--parent-checkpoint", eval_args.parent_checkpoint,
            "--out-dir", eval_args.out_dir]
if eval_args.headless:
    sys.argv.append("--headless")
import learning.rl.gate_f10h_residual_ppo_smoke as smoke  # noqa: E402
sys.argv = saved_argv

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.models.mlp_model import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from learning.rl.env.planar_roi_diagnostics import surface_view  # noqa: E402
from learning.rl.gate_e6_area_quality import (  # noqa: E402
    AREA_DIAGNOSTIC_VERSION, AreaQualityTargets, summarize_area_quality)


AREA_KEYS = ("gu_pass", "ra_pass", "rz_pass", "scratch_pass", "all4_pass",
             "clearcoat_pass", "temperature_pass")


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def area_columns(summary, prefix):
    row = {}
    for key in AREA_KEYS:
        row[f"{prefix}_{key}_area_pct"] = summary[f"roi_{key}_area_pct"]
        row[f"{prefix}_tile_{key}_area_pct_mean"] = summary[f"tile_{key}_area_pct_mean"]
        row[f"{prefix}_tile_{key}_area_pct_min"] = summary[f"tile_{key}_area_pct_min"]
        row[f"{prefix}_tile_{key}_area_pct_p10"] = summary[f"tile_{key}_area_pct_p10"]
    return row


def load_residual_actor(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["actor_state_dict"]
    if tuple(state["mlp.0.weight"].shape) != (128, 20):
        raise ValueError("residual checkpoint must contain a 20-D actor")
    dummy = TensorDict({"policy": torch.zeros(1, 20, device=device)}, batch_size=[1])
    actor = MLPModel(
        dummy, {"actor": ["policy"]}, "actor", 2,
        hidden_dims=[128, 128], activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3,
                          "std_type": "scalar"}).to(device)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor, int(checkpoint.get("iter", -1))


class ResidualEvalEnv(smoke.ResidualPPOEnv):
    control_parent = False

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.final_area = {}
        self.latched_safety = {}
        super().__init__(cfg, render_mode, **kwargs)

    def capture_final(self, env_id):
        roi = surface_view(self._surfaces[env_id], self._roi_slices)
        targets = AreaQualityTargets(
            ra_max_um=float(self.cfg.t_ra_pass_max_um),
            rz_max_um=float(self.cfg.t_rz_pass_max_um),
            clearcoat_min_um=float(self.cfg.clearcoat_safety_limit_um),
            temperature_max_c=float(self.cfg.thermal_hard_limit_c))
        self.final_area[env_id] = summarize_area_quality(roi, targets)
        self.latched_safety[env_id] = {
            "force_hard_violated": bool(self._force_hard_violated[env_id]),
            "thermal_hard_violated": bool(self._thermal_hard_violated[env_id]),
            "unstable_hard_violated": bool(self._unstable_hard_violated[env_id]),
        }

    def _repolish_decide(self, env_id):
        terminate = super()._repolish_decide(env_id)
        if terminate:
            self.capture_final(env_id)
        return terminate

    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        for env_id in (terminated | truncated).nonzero(
                as_tuple=False).squeeze(-1).cpu().tolist():
            if env_id not in self.final_area:
                self.capture_final(env_id)
        return terminated, truncated

    def _pre_physics_step(self, actions):
        if self.control_parent:
            return smoke.GateF7CurvedObservationEnv._pre_physics_step(self, actions)
        return super()._pre_physics_step(actions)

    def _apply_action(self):
        if self.control_parent:
            return smoke.GateF7CurvedObservationEnv._apply_action(self)
        return super()._apply_action()


def make_cfg():
    cfg = smoke.GateF7CurvedObservationEnvCfg()
    cfg.factory_profile_ids = smoke.PROFILE_IDS
    cfg.factory_envs_per_profile = eval_args.envs_per_profile
    cfg.scene.num_envs = len(smoke.PROFILE_IDS) * eval_args.envs_per_profile
    cfg.factory_path_mode = "gate_c"
    cfg.factory_gate_c_direction_mode = eval_args.direction_mode
    cfg.surface_kind = eval_args.surface_kind
    cfg.curvature_radius_m = 0.60
    cfg.align_pad_to_surface_normal = True
    cfg.gate_f7_observation_mode = smoke.NORMAL_CURVATURE20
    cfg.observation_space = smoke.observation_dim(smoke.NORMAL_CURVATURE20)
    cfg.surface_seed_base = eval_args.surface_seed_base
    cfg.seed = eval_args.physics_seed
    cfg.enable_pad_physical_contact = True
    cfg.repolish_max_passes = 1
    cfg.episode_length_s = 840.0
    return cfg


def main():
    parent = os.path.abspath(eval_args.parent_checkpoint)
    residual = (os.path.abspath(eval_args.residual_checkpoint)
                if eval_args.residual_checkpoint else None)
    out_dir = os.path.abspath(eval_args.out_dir)
    if not os.path.isfile(parent):
        raise FileNotFoundError(parent)
    if eval_args.arm == "residual_ppo" and (not residual or not os.path.isfile(residual)):
        raise FileNotFoundError(residual or "--residual-checkpoint is required")
    if os.path.exists(out_dir):
        raise FileExistsError(out_dir)
    os.makedirs(out_dir)
    cfg = make_cfg()
    ResidualEvalEnv.parent_checkpoint = parent
    ResidualEvalEnv.control_parent = eval_args.arm == "control_parent"
    env = ResidualEvalEnv(cfg, render_mode=None)
    env._repolish_mode = True
    actor = None
    saved_iteration = -1
    if eval_args.arm == "residual_ppo":
        actor, saved_iteration = load_residual_actor(residual, str(env.device))
    torch.manual_seed(eval_args.policy_seed)
    obs, _ = env.reset()
    seeds = np.asarray([int(env._factory_metadata[i]["profile_seed"])
                        for i in range(env.num_envs)], dtype=np.int64)
    targets = AreaQualityTargets(
        ra_max_um=float(cfg.t_ra_pass_max_um), rz_max_um=float(cfg.t_rz_pass_max_um),
        clearcoat_min_um=float(cfg.clearcoat_safety_limit_um),
        temperature_max_c=float(cfg.thermal_hard_limit_c))
    initial_area = {i: summarize_area_quality(
        surface_view(env._surfaces[i], env._roi_slices), targets)
        for i in range(env.num_envs)}
    active = set(range(env.num_envs))
    steps = np.zeros(env.num_envs, dtype=np.int64)
    contact_steps = np.zeros(env.num_envs, dtype=np.int64)
    sensor_fault_steps = np.zeros(env.num_envs, dtype=np.int64)
    raw_force_max = np.zeros(env.num_envs)
    residual_abs_sum = np.zeros(env.num_envs)
    sequence_rows, tile_rows = [], []
    try:
        for step in range(eval_args.max_control_steps):
            if not active:
                break
            full = obs["policy"]
            if full.shape != (env.num_envs, 20) or not bool(torch.isfinite(full).all()):
                raise RuntimeError("invalid F10-H normal_curvature20 observation")
            if eval_args.arm == "control_parent":
                with torch.no_grad():
                    actions = env._f10h_parent(TensorDict(
                        {"policy": full[:, :14]}, batch_size=[env.num_envs])).clamp(-1.0, 1.0)
            elif actor is None:
                actions = torch.zeros(env.num_envs, 2, device=env.device)
            else:
                with torch.no_grad():
                    actions = actor(TensorDict(
                        {"policy": full}, batch_size=[env.num_envs])).clamp(-1.0, 1.0)
            inactive = sorted(set(range(env.num_envs)) - active)
            if inactive:
                actions[inactive] = 0.0
            action_np = actions.detach().cpu().numpy()
            for i in active:
                steps[i] += 1
                residual_abs_sum[i] += float(np.abs(action_np[i]).mean())
            obs, _, terminated, truncated, _ = env.step(actions)
            used = env._force_used_n.detach().cpu().numpy()
            faults = env._sensor_fault.detach().cpu().numpy()
            raw = env._force_sensor_n.detach().cpu().numpy()
            for i in active:
                contact_steps[i] += int(used[i] > 0.05)
                sensor_fault_steps[i] += int(faults[i])
                raw_force_max[i] = max(raw_force_max[i], float(raw[i]))
            done_ids = (terminated | truncated).nonzero(
                as_tuple=False).squeeze(-1).cpu().tolist()
            for i in done_ids:
                if i not in active:
                    continue
                log = env._repolish_log.pop(i, None)
                final_area = env.final_area.pop(i, None)
                safety = env.latched_safety.pop(i, None)
                if log is None or final_area is None or safety is None:
                    raise RuntimeError(f"missing final evidence for env {i}")
                n = max(1, int(steps[i]))
                profile = str(env._factory_metadata[i]["profile_id"])
                before, final = log["before"], log["final"]
                sequence_rows.append({
                    "arm": eval_args.arm, "surface_kind": eval_args.surface_kind,
                    "direction_mode": eval_args.direction_mode,
                    "surface_profile": profile, "env": i,
                    "profile_seed": int(seeds[i]), "outcome": log["outcome"],
                    "quality_ok": bool(log["quality_ok"]),
                    "safety_ok": bool(log["safety_ok"]), **safety,
                    "control_steps": n, "contact_steps": int(contact_steps[i]),
                    "sensor_fault_steps": int(sensor_fault_steps[i]),
                    "raw_force_max_n": float(raw_force_max[i]),
                    "raw_residual_action_abs_mean": float(residual_abs_sum[i] / n),
                    "gu_before": before["gu"], "gu_final": final["gu"],
                    "ra_before_um": before["ra"], "ra_final_um": final["ra"],
                    "rz_before_um": before["rz"], "rz_final_um": final["rz"],
                    "scratch_before_um": before["scratch"],
                    "scratch_final_um": final["scratch"],
                    "clearcoat_min_um": final["cc_min"],
                    "temperature_peak_c": final["temperature_peak_c"],
                    **area_columns(initial_area[i]["summary"], "before"),
                    **area_columns(final_area["summary"], "final"),
                })
                for tile in final_area["tile_rows"]:
                    tile_rows.append({
                        "arm": eval_args.arm, "surface_kind": eval_args.surface_kind,
                        "direction_mode": eval_args.direction_mode,
                        "surface_profile": profile, "env": i,
                        "profile_seed": int(seeds[i]), **tile})
                active.remove(i)
            if (step + 1) % 1000 == 0:
                print(f"[F10-H eval] {eval_args.arm}/{eval_args.surface_kind}/"
                      f"{eval_args.direction_mode} step={step+1} "
                      f"complete={env.num_envs-len(active)}/{env.num_envs}", flush=True)
        if eval_args.smoke_only:
            smoke_result = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "gate": "F10H_FIXED_EVALUATION_SMOKE",
                "arm": eval_args.arm,
                "surface_kind": eval_args.surface_kind,
                "steps": eval_args.max_control_steps,
                "active_envs_after": len(active),
                "observations_and_actions_finite": True,
                "cap_contract_failures": env._f10h_cap_contract_failures,
                "supervisor_faults": env._f10h_supervisor_faults,
                "force_overload_seen": env._f10h_force_overload_seen,
                "sensor_faults_current": int(env._sensor_fault.sum().item()),
            }
            smoke_result["smoke_pass"] = all(smoke_result[name] == 0 for name in (
                "cap_contract_failures", "supervisor_faults", "force_overload_seen",
                "sensor_faults_current"))
            with open(os.path.join(out_dir, "smoke.json"), "w", encoding="utf-8") as fh:
                json.dump(smoke_result, fh, indent=2, sort_keys=True)
            print(json.dumps(smoke_result, indent=2, sort_keys=True), flush=True)
            return
        write_csv(os.path.join(out_dir, "sequences.csv"), sequence_rows)
        write_csv(os.path.join(out_dir, "tile_area_fractions.csv"), tile_rows)
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "gate": "F10H_FIXED_EVALUATION", "arm": eval_args.arm,
            "surface_kind": eval_args.surface_kind,
            "direction_mode": eval_args.direction_mode,
            "profiles": smoke.PROFILE_IDS,
            "envs_per_profile": eval_args.envs_per_profile,
            "surface_seed_base": eval_args.surface_seed_base,
            "physics_seed": eval_args.physics_seed,
            "policy_seed": eval_args.policy_seed,
            "parent_checkpoint": parent,
            "parent_checkpoint_sha256": smoke.sha256(parent),
            "residual_checkpoint": residual,
            "residual_checkpoint_sha256": smoke.sha256(residual) if residual else None,
            "saved_iteration": saved_iteration,
            "observation_mode": smoke.NORMAL_CURVATURE20,
            "safety_interface": ("FROZEN_PARENT_NO_SHIELD"
                                 if eval_args.arm == "control_parent"
                                 else "F10H_V2_BOUNDED_RESIDUAL_G3_SUBSTEP"),
            "diagnostic_version": AREA_DIAGNOSTIC_VERSION,
            "path": dict(env.factory_path_metadata),
            "completed": len(sequence_rows), "expected": env.num_envs,
            "tile_rows": len(tile_rows),
            "cap_contract_failures": env._f10h_cap_contract_failures,
            "supervisor_faults": env._f10h_supervisor_faults,
            "force_overload_seen": env._f10h_force_overload_seen,
            "supervisor_interventions": env._f10h_supervisor_interventions,
            "all_finite": all(np.isfinite(value) for row in sequence_rows
                              for value in row.values()
                              if isinstance(value, (int, float)) and not isinstance(value, bool)),
            "training_performed": False,
        }
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)
        print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
        if active:
            raise RuntimeError(f"incomplete F10-H evaluation: {len(sequence_rows)}/{env.num_envs}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        smoke.app.close()
