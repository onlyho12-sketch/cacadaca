"""Gate F10-G curvature-conditioned deterministic PhysX pilot (no training)."""
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
parser.add_argument("--shield-mode", choices=("control", "curvature_safety"), required=True)
parser.add_argument("--surface-kind", choices=("flat", "cylinder", "sphere", "freeform"), required=True)
parser.add_argument("--curvature-radius-m", type=float, default=0.60)
parser.add_argument("--direction-mode", choices=("same_xx", "cross_xy"), required=True)
parser.add_argument("--envs-per-profile", type=int, default=1)
parser.add_argument("--surface-seed-base", type=int, default=33000)
parser.add_argument("--physics-seed", type=int, default=20260901)
parser.add_argument("--policy-seed", type=int, default=20260831)
parser.add_argument("--max-control-steps", type=int, default=8000)
parser.add_argument("--smoke-only", action="store_true")
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
from learning.rl.env.planar_roi_diagnostics import surface_view  # noqa: E402
from learning.rl.gate_e6_area_quality import (  # noqa: E402
    AREA_DIAGNOSTIC_VERSION, AreaQualityTargets, summarize_area_quality)
from learning.rl.gate_f7_observation import BASE14, observation_dim  # noqa: E402
from learning.rl.gate_f10_curvature_safety import (  # noqa: E402
    SafetyConfig, apply_curvature_physx_action_shield)


AREA_KEYS = ("gu_pass", "ra_pass", "rz_pass", "scratch_pass", "all4_pass",
             "clearcoat_pass", "temperature_pass")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _load_policy(checkpoint: str, device: str):
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ck["actor_state_dict"]
    if tuple(state["mlp.0.weight"].shape) != (128, 14):
        raise ValueError("F8 parent must be a 14-D actor")
    dummy = TensorDict({"policy": torch.zeros(1, 14, device=device)}, batch_size=[1])
    actor = MLPModel(
        dummy, {"actor": ["policy"]}, "actor", 2,
        hidden_dims=[128, 128], activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3,
                          "std_type": "scalar"}).to(device)
    actor.load_state_dict(state, strict=True); actor.eval()
    return actor, int(ck.get("iter", -1))


class GateF8EvalEnv(GateF7CurvedObservationEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        self.gate_f8_final_area = {}
        self.gate_f8_latched_safety = {}
        super().__init__(cfg, render_mode, **kwargs)

    def _capture_final(self, env_id: int) -> None:
        roi = surface_view(self._surfaces[env_id], self._roi_slices)
        targets = AreaQualityTargets(
            ra_max_um=float(self.cfg.t_ra_pass_max_um),
            rz_max_um=float(self.cfg.t_rz_pass_max_um),
            clearcoat_min_um=float(self.cfg.clearcoat_safety_limit_um),
            temperature_max_c=float(self.cfg.thermal_hard_limit_c))
        self.gate_f8_final_area[env_id] = summarize_area_quality(roi, targets)
        self.gate_f8_latched_safety[env_id] = {
            "force_hard_violated": bool(self._force_hard_violated[env_id]),
            "thermal_hard_violated": bool(self._thermal_hard_violated[env_id]),
            "unstable_hard_violated": bool(self._unstable_hard_violated[env_id])}

    def _repolish_decide(self, env_id: int) -> bool:
        terminate = super()._repolish_decide(env_id)
        if terminate:
            self._capture_final(env_id)
        return terminate

    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        for env_id in (terminated | truncated).nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
            if env_id not in self.gate_f8_final_area:
                self._capture_final(env_id)
        return terminated, truncated


def _area_columns(summary: dict, prefix: str) -> dict:
    row = {}
    for key in AREA_KEYS:
        row[f"{prefix}_{key}_area_pct"] = summary[f"roi_{key}_area_pct"]
        row[f"{prefix}_tile_{key}_area_pct_mean"] = summary[f"tile_{key}_area_pct_mean"]
        row[f"{prefix}_tile_{key}_area_pct_min"] = summary[f"tile_{key}_area_pct_min"]
        row[f"{prefix}_tile_{key}_area_pct_p10"] = summary[f"tile_{key}_area_pct_p10"]
    return row


def main() -> None:
    checkpoint = os.path.abspath(args.checkpoint)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)
    cfg = GateF7CurvedObservationEnvCfg()
    cfg.factory_profile_ids = PROFILE_IDS
    cfg.factory_envs_per_profile = args.envs_per_profile
    cfg.scene.num_envs = len(PROFILE_IDS) * args.envs_per_profile
    cfg.factory_path_mode = "gate_c"
    cfg.factory_gate_c_direction_mode = args.direction_mode
    cfg.surface_kind = args.surface_kind
    cfg.curvature_radius_m = args.curvature_radius_m
    cfg.align_pad_to_surface_normal = True
    cfg.gate_f7_observation_mode = BASE14
    cfg.observation_space = observation_dim(BASE14)
    cfg.surface_seed_base = args.surface_seed_base
    cfg.seed = args.physics_seed
    cfg.enable_pad_physical_contact = True
    cfg.repolish_max_passes = 1
    cfg.episode_length_s = 840.0
    env = GateF8EvalEnv(cfg, render_mode=None)
    env._repolish_mode = True
    actor, saved_iteration = _load_policy(checkpoint, str(env.device))
    torch.manual_seed(args.policy_seed)
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
    parent_sum = np.zeros((env.num_envs, 2)); executed_sum = np.zeros((env.num_envs, 2))
    cap_sum = np.zeros(env.num_envs); cap_min = np.ones(env.num_envs)
    shield_steps = np.zeros(env.num_envs, dtype=np.int64)
    control_steps = np.zeros(env.num_envs, dtype=np.int64)
    contact_steps = np.zeros(env.num_envs, dtype=np.int64)
    sensor_fault_steps = np.zeros(env.num_envs, dtype=np.int64)
    raw_force_max = np.zeros(env.num_envs); align_max = np.zeros(env.num_envs)
    risk_sum = np.zeros(env.num_envs); risk_max = np.zeros(env.num_envs)
    force_cmd_jump_max = np.zeros(env.num_envs); feed_cmd_jump_max = np.zeros(env.num_envs)
    flat_parity_all = np.ones(env.num_envs, dtype=bool)
    previous_force = None; previous_feed = None
    safety_cfg = SafetyConfig()
    sequence_rows, tile_rows = [], []
    try:
        for step in range(args.max_control_steps):
            if not active:
                break
            full = obs["policy"]
            if full.shape != (env.num_envs, 14) or not bool(torch.isfinite(full).all()):
                raise RuntimeError("invalid F8 base14 observation")
            with torch.no_grad():
                td = TensorDict({"policy": full}, batch_size=[env.num_envs])
                parent_action = actor(td).clamp(-1.0, 1.0)
            parent_np = parent_action.detach().cpu().numpy()
            if args.shield_mode == "control":
                executed_np = parent_np.copy()
                force_cmd = float(env.recipe.target_contact_force_n) * (
                    1.0 + executed_np[:, 0] * float(cfg.force_ratio_limit))
                feed_cmd = float(env.recipe.feed_speed_mm_s) * (
                    1.0 + executed_np[:, 1] * float(cfg.feed_ratio_limit))
                shield = {"risk": np.zeros(env.num_envs),
                          "force_cap_action": np.ones(env.num_envs),
                          "flat_exact_parity": np.ones(env.num_envs, dtype=bool)}
            else:
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
                force_cmd = shield["force_command_n"]
                feed_cmd = shield["feed_command_mm_s"]
            caps = shield["force_cap_action"]
            if previous_force is not None:
                force_cmd_jump_max = np.maximum(force_cmd_jump_max,
                                                np.abs(force_cmd - previous_force))
                feed_cmd_jump_max = np.maximum(feed_cmd_jump_max,
                                               np.abs(feed_cmd - previous_feed))
            previous_force = np.asarray(force_cmd).copy()
            previous_feed = np.asarray(feed_cmd).copy()
            executed = torch.as_tensor(executed_np, device=env.device)
            inactive = sorted(set(range(env.num_envs)) - active)
            if inactive:
                executed[inactive] = 0.0
            for i in active:
                p = parent_action[i].detach().cpu().numpy()
                parent_sum[i] += p; executed_sum[i] += executed_np[i]
                cap_sum[i] += caps[i]; cap_min[i] = min(cap_min[i], caps[i])
                shield_steps[i] += int(np.any(np.abs(executed_np[i] - p) > 1e-7))
                risk_sum[i] += float(shield["risk"][i])
                risk_max[i] = max(risk_max[i], float(shield["risk"][i]))
                if args.surface_kind == "flat":
                    flat_parity_all[i] &= bool(np.array_equal(executed_np[i], p))
                control_steps[i] += 1
            obs, _, terminated, truncated, _ = env.step(executed)
            used = env._force_used_n.detach().cpu().numpy()
            faults = env._sensor_fault.detach().cpu().numpy()
            raw = env._force_sensor_n.detach().cpu().numpy()
            align = env._normal_alignment_error_deg.detach().cpu().numpy()
            for i in active:
                contact_steps[i] += int(used[i] > 0.05); sensor_fault_steps[i] += int(faults[i])
                raw_force_max[i] = max(raw_force_max[i], float(raw[i]))
                align_max[i] = max(align_max[i], float(align[i]))
            for i in (terminated | truncated).nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
                if i not in active:
                    continue
                log = env._repolish_log.pop(i, None)
                final_area = env.gate_f8_final_area.pop(i, None)
                latched = env.gate_f8_latched_safety.pop(i, None)
                if log is None or final_area is None or latched is None:
                    raise RuntimeError(f"missing F8 final evidence for env {i}")
                n = max(1, int(control_steps[i])); profile = str(env._factory_metadata[i]["profile_id"])
                before, final = log["before"], log["final"]
                sequence_rows.append({
                    "shield_mode": args.shield_mode, "surface_kind": args.surface_kind,
                    "direction_mode": args.direction_mode, "surface_profile": profile,
                    "env": i, "profile_seed": int(seeds[i]), "outcome": log["outcome"],
                    "quality_ok": bool(log["quality_ok"]), "safety_ok": bool(log["safety_ok"]),
                    **latched, "control_steps": n, "contact_steps": int(contact_steps[i]),
                    "sensor_fault_steps": int(sensor_fault_steps[i]),
                    "raw_force_max_n": float(raw_force_max[i]),
                    "normal_alignment_error_max_deg": float(align_max[i]),
                    "parent_force_action_mean": float(parent_sum[i, 0] / n),
                    "executed_force_action_mean": float(executed_sum[i, 0] / n),
                    "parent_feed_action_mean": float(parent_sum[i, 1] / n),
                    "shield_step_fraction": float(shield_steps[i] / n),
                    "geometry_risk_mean": float(risk_sum[i] / n),
                    "geometry_risk_max": float(risk_max[i]),
                    "force_command_jump_max_n": float(force_cmd_jump_max[i]),
                    "feed_command_jump_max_mm_s": float(feed_cmd_jump_max[i]),
                    "flat_action_exact_parity": bool(flat_parity_all[i]),
                    "force_action_cap_mean": float(cap_sum[i] / n),
                    "force_action_cap_min": float(cap_min[i]),
                    "gu_before": before["gu"], "gu_final": final["gu"],
                    "ra_before_um": before["ra"], "ra_final_um": final["ra"],
                    "rz_before_um": before["rz"], "rz_final_um": final["rz"],
                    "scratch_before_um": before["scratch"], "scratch_final_um": final["scratch"],
                    "clearcoat_min_um": final["cc_min"], "temperature_peak_c": final["temperature_peak_c"],
                    **_area_columns(initial_area[i]["summary"], "before"),
                    **_area_columns(final_area["summary"], "final")})
                for tile in final_area["tile_rows"]:
                    tile_rows.append({"shield_mode": args.shield_mode,
                                      "surface_kind": args.surface_kind,
                                      "direction_mode": args.direction_mode,
                                      "surface_profile": profile, "env": i,
                                      "profile_seed": int(seeds[i]), **tile})
                active.remove(i)
            if (step + 1) % 1000 == 0:
                print(f"[Gate F10-G] {args.shield_mode}/{args.surface_kind}/{args.direction_mode} "
                      f"step={step + 1} complete={env.num_envs-len(active)}/{env.num_envs}", flush=True)
        if args.smoke_only:
            smoke = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "gate": "F10G_CURVATURE_SAFETY_SMOKE",
                "shield_mode": args.shield_mode,
                "surface_kind": args.surface_kind,
                "steps": args.max_control_steps,
                "active_envs_after": len(active),
                "all_observations_and_actions_finite": True,
                "sensor_fault_steps": int(sensor_fault_steps.sum()),
                "smoke_pass": int(sensor_fault_steps.sum()) == 0,
                "training_performed": False,
            }
            with open(os.path.join(out_dir, "smoke.json"), "w", encoding="utf-8") as handle:
                json.dump(smoke, handle, indent=2, sort_keys=True)
            print(json.dumps(smoke, indent=2, sort_keys=True), flush=True)
            return
        _write_csv(os.path.join(out_dir, "sequences.csv"), sequence_rows)
        _write_csv(os.path.join(out_dir, "tile_area_fractions.csv"), tile_rows)
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(), "gate": "F10G_CURVATURE_SAFETY",
            "shield_mode": args.shield_mode, "surface_kind": args.surface_kind,
            "direction_mode": args.direction_mode, "profiles": PROFILE_IDS,
            "curvature_radius_m": args.curvature_radius_m,
            "envs_per_profile": args.envs_per_profile, "surface_seed_base": args.surface_seed_base,
            "physics_seed": args.physics_seed, "policy_seed": args.policy_seed,
            "checkpoint": checkpoint, "checkpoint_sha256": _sha256(checkpoint),
            "saved_iteration": saved_iteration, "observation_mode": BASE14,
            "safety_interface": "F10F_CURVATURE_SAFETY",
            "path": dict(env.factory_path_metadata), "diagnostic_version": AREA_DIAGNOSTIC_VERSION,
            "completed": len(sequence_rows), "expected": env.num_envs,
            "tile_rows": len(tile_rows), "training_performed": False,
            "all_finite": all(np.isfinite(v) for row in sequence_rows for v in row.values()
                              if isinstance(v, (int, float)) and not isinstance(v, bool))}
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
        if active:
            raise RuntimeError(f"incomplete F8 evaluation: {len(sequence_rows)}/{env.num_envs}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
