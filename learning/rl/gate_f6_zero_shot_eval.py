"""Gate F6 zero-shot evaluation of the frozen flat E7 release on curved PhysX."""
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
parser.add_argument("--surface-kind", choices=("flat", "cylinder", "sphere", "freeform"),
                    required=True)
parser.add_argument("--radius-m", type=float, default=0.60)
parser.add_argument("--freeform-seed", type=int, default=0)
parser.add_argument("--direction-mode", choices=("same_xx", "cross_xy"), required=True)
parser.add_argument("--envs-per-profile", type=int, default=2)
parser.add_argument("--surface-seed-base", type=int, default=31000)
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
from learning.polytwin.gloss_proxy import LiteratureGlossProxyModel  # noqa: E402
from learning.rl.env.gate_e_mixed_observation_env import GateEMixedObservationEnv  # noqa: E402
from learning.rl.env.gate_f_curved_polish_env import GateFCurvedPolishEnv  # noqa: E402
from learning.rl.env.gate_f_curved_polish_env_cfg import GateFCurvedPolishEnvCfg  # noqa: E402
from learning.rl.env.planar_roi_diagnostics import surface_view  # noqa: E402
from learning.rl.env.planar_roi_polish_env import PlanarRoiPolishEnv  # noqa: E402
from learning.rl.gate_f_curved_geometry import SurfaceGeometrySpec  # noqa: E402
from learning.rl.gate_e6_area_quality import (  # noqa: E402
    AREA_DIAGNOSTIC_VERSION,
    AreaQualityTargets,
    summarize_area_quality,
)


AREA_KEYS = (
    "gu_pass", "ra_pass", "rz_pass", "scratch_pass", "all4_pass",
    "clearcoat_pass", "temperature_pass",
)


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
        writer.writeheader()
        writer.writerows(rows)


def _load_policy(checkpoint: str, device: str):
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ck["actor_state_dict"]
    obs_dim = int(state["mlp.0.weight"].shape[1])
    action_dim = int(state["mlp.4.weight"].shape[0])
    if (obs_dim, action_dim) != (14, 2):
        raise ValueError(f"F6 release must be 14->2, got {obs_dim}->{action_dim}")
    dummy = TensorDict({"policy": torch.zeros(1, obs_dim, device=device)}, batch_size=[1])
    actor = MLPModel(
        dummy, {"actor": ["policy"]}, "actor", action_dim,
        hidden_dims=[128, 128], activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3,
                          "std_type": "scalar"},
    ).to(device)
    actor.load_state_dict(state, strict=True)
    actor.eval()

    def policy(observation: torch.Tensor) -> torch.Tensor:
        td = TensorDict({"policy": observation[:, :obs_dim]}, batch_size=[len(observation)])
        return actor(td).clamp(-1.0, 1.0)

    return policy, obs_dim, action_dim, int(ck.get("iter", -1))


class GateF6ZeroShotEnv(GateFCurvedPolishEnv):
    """Gate F geometry plus the full frozen Gate E profile set and area capture."""

    def __init__(self, cfg: GateFCurvedPolishEnvCfg, render_mode=None, **kwargs):
        profiles = tuple(cfg.factory_profile_ids)
        if not profiles or any(profile not in PROFILE_IDS for profile in profiles):
            raise ValueError(f"F6 profiles must be drawn from {PROFILE_IDS}")
        if len(set(profiles)) != len(profiles):
            raise ValueError("F6 profiles must be distinct")
        expected = len(profiles) * int(cfg.factory_envs_per_profile)
        if cfg.scene.num_envs != expected:
            raise ValueError(f"F6 expected {expected} envs, got {cfg.scene.num_envs}")
        self._geometry_spec = SurfaceGeometrySpec(
            kind=str(cfg.surface_kind), patch_size_m=tuple(cfg.patch_size_m),
            curvature_radius_m=float(cfg.curvature_radius_m),
            freeform_seed=int(cfg.freeform_seed),
        )
        self._geometry_spec.validate()
        self._factory_metadata = {}
        self._factory_gloss = LiteratureGlossProxyModel()
        self.gate_f6_final_area: dict[int, dict] = {}
        self.gate_f6_latched_safety: dict[int, dict] = {}
        PlanarRoiPolishEnv.__init__(self, cfg, render_mode, **kwargs)

    def factory_diagnostic(self, env_id: int) -> dict:
        return GateEMixedObservationEnv.factory_diagnostic(self, env_id)

    def _capture_final(self, env_id: int) -> None:
        roi = surface_view(self._surfaces[env_id], self._roi_slices)
        targets = AreaQualityTargets(
            ra_max_um=float(self.cfg.t_ra_pass_max_um),
            rz_max_um=float(self.cfg.t_rz_pass_max_um),
            clearcoat_min_um=float(self.cfg.clearcoat_safety_limit_um),
            temperature_max_c=float(self.cfg.thermal_hard_limit_c),
        )
        self.gate_f6_final_area[env_id] = summarize_area_quality(roi, targets)
        self.gate_f6_latched_safety[env_id] = {
            "force_hard_violated": bool(self._force_hard_violated[env_id]),
            "thermal_hard_violated": bool(self._thermal_hard_violated[env_id]),
            "unstable_hard_violated": bool(self._unstable_hard_violated[env_id]),
        }

    def _repolish_decide(self, env_id: int) -> bool:
        terminate = super()._repolish_decide(env_id)
        if terminate:
            self._capture_final(env_id)
        return terminate

    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        done_ids = (terminated | truncated).nonzero(
            as_tuple=False).squeeze(-1).cpu().tolist()
        for env_id in done_ids:
            if env_id not in self.gate_f6_final_area:
                self._capture_final(env_id)
        return terminated, truncated


def _area_columns(summary: dict, prefix: str) -> dict:
    row = {}
    for key in AREA_KEYS:
        row[f"{prefix}_{key}_area_pct"] = summary[f"roi_{key}_area_pct"]
        row[f"{prefix}_tile_{key}_area_pct_mean"] = summary[
            f"tile_{key}_area_pct_mean"]
        row[f"{prefix}_tile_{key}_area_pct_min"] = summary[
            f"tile_{key}_area_pct_min"]
        row[f"{prefix}_tile_{key}_area_pct_p10"] = summary[
            f"tile_{key}_area_pct_p10"]
    return row


def main() -> None:
    checkpoint = os.path.abspath(args.checkpoint)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)

    cfg = GateFCurvedPolishEnvCfg()
    cfg.factory_profile_ids = PROFILE_IDS
    cfg.factory_envs_per_profile = int(args.envs_per_profile)
    cfg.scene.num_envs = len(PROFILE_IDS) * int(args.envs_per_profile)
    cfg.factory_path_mode = "gate_c"
    cfg.factory_gate_c_step_over_ratio = 0.40
    cfg.factory_gate_c_edge_mode = "balanced_extend5"
    cfg.factory_gate_c_direction_mode = args.direction_mode
    cfg.surface_kind = args.surface_kind
    cfg.curvature_radius_m = float(args.radius_m)
    cfg.freeform_seed = int(args.freeform_seed)
    cfg.align_pad_to_surface_normal = True
    cfg.surface_seed_base = int(args.surface_seed_base)
    cfg.seed = int(args.physics_seed)
    cfg.robot_feed_speed_mm_s = 12.7
    cfg.enable_pad_physical_contact = True
    cfg.repolish_max_passes = 1
    cfg.repolish_cooldown_s = 20.0
    cfg.episode_length_s = 840.0

    env = GateF6ZeroShotEnv(cfg, render_mode=None)
    env._repolish_mode = True
    policy, obs_dim, action_dim, saved_iteration = _load_policy(checkpoint, str(env.device))
    obs, _ = env.reset()
    if obs["policy"].shape[1] != obs_dim:
        raise RuntimeError(
            f"F6 environment observation {obs['policy'].shape[1]} != release {obs_dim}")
    torch.manual_seed(int(args.policy_seed))

    seeds = np.asarray([
        int(env._factory_metadata[i]["profile_seed"]) for i in range(env.num_envs)
    ], dtype=np.int64)
    targets = AreaQualityTargets(
        ra_max_um=float(cfg.t_ra_pass_max_um), rz_max_um=float(cfg.t_rz_pass_max_um),
        clearcoat_min_um=float(cfg.clearcoat_safety_limit_um),
        temperature_max_c=float(cfg.thermal_hard_limit_c),
    )
    initial_area = {
        i: summarize_area_quality(surface_view(env._surfaces[i], env._roi_slices), targets)
        for i in range(env.num_envs)
    }
    active = set(range(env.num_envs))
    action_sum = np.zeros((env.num_envs, 2), dtype=np.float64)
    control_steps = np.zeros(env.num_envs, dtype=np.int64)
    contact_steps = np.zeros(env.num_envs, dtype=np.int64)
    sensor_fault_steps = np.zeros(env.num_envs, dtype=np.int64)
    raw_force_max = np.zeros(env.num_envs, dtype=np.float64)
    alignment_error_max = np.zeros(env.num_envs, dtype=np.float64)
    sequence_rows: list[dict] = []
    tile_rows: list[dict] = []
    step = 0
    try:
        while active and step < int(args.max_control_steps):
            full = obs["policy"]
            if full.shape[1] != obs_dim or not bool(torch.isfinite(full).all()):
                raise RuntimeError("F6 observation shape/finiteness failure")
            with torch.no_grad():
                actions = policy(full)
            if not bool(torch.isfinite(actions).all()):
                raise RuntimeError("F6 policy produced NaN/Inf")
            inactive = sorted(set(range(env.num_envs)) - active)
            if inactive:
                actions[inactive] = 0.0
            for env_id in active:
                action_sum[env_id] += actions[env_id].detach().cpu().numpy()
                control_steps[env_id] += 1
            obs, _, terminated, truncated, _ = env.step(actions)
            step += 1
            used = env._force_used_n.detach().cpu().numpy()
            faults = env._sensor_fault.detach().cpu().numpy()
            raw = env._force_sensor_n.detach().cpu().numpy()
            align = env._normal_alignment_error_deg.detach().cpu().numpy()
            for env_id in active:
                contact_steps[env_id] += int(used[env_id] > 0.05)
                sensor_fault_steps[env_id] += int(faults[env_id])
                raw_force_max[env_id] = max(raw_force_max[env_id], float(raw[env_id]))
                alignment_error_max[env_id] = max(
                    alignment_error_max[env_id], float(align[env_id]))
            done_ids = (terminated | truncated).nonzero(
                as_tuple=False).squeeze(-1).cpu().tolist()
            for env_id in done_ids:
                if env_id not in active:
                    continue
                log = env._repolish_log.pop(env_id, None)
                final_area = env.gate_f6_final_area.pop(env_id, None)
                latched = env.gate_f6_latched_safety.pop(env_id, None)
                if log is None or final_area is None or latched is None:
                    raise RuntimeError(f"missing F6 final evidence for env {env_id}")
                profile = str(env._factory_metadata[env_id]["profile_id"])
                before, final = log["before"], log["final"]
                n = max(1, int(control_steps[env_id]))
                sequence_rows.append({
                    "surface_kind": args.surface_kind,
                    "curvature_radius_m": float(args.radius_m),
                    "freeform_seed": int(args.freeform_seed),
                    "direction_mode": args.direction_mode,
                    "surface_profile": profile,
                    "env": env_id,
                    "profile_seed": int(seeds[env_id]),
                    "outcome": log["outcome"],
                    "quality_ok": bool(log["quality_ok"]),
                    "safety_ok": bool(log["safety_ok"]),
                    **latched,
                    "control_steps": n,
                    "contact_steps": int(contact_steps[env_id]),
                    "sensor_fault_steps": int(sensor_fault_steps[env_id]),
                    "raw_force_max_n": float(raw_force_max[env_id]),
                    "normal_alignment_error_max_deg": float(alignment_error_max[env_id]),
                    "force_action_mean": float(action_sum[env_id, 0] / n),
                    "feed_action_mean": float(action_sum[env_id, 1] / n),
                    "gu_before": before["gu"], "gu_final": final["gu"],
                    "ra_before_um": before["ra"], "ra_final_um": final["ra"],
                    "rz_before_um": before["rz"], "rz_final_um": final["rz"],
                    "scratch_before_um": before["scratch"],
                    "scratch_final_um": final["scratch"],
                    "clearcoat_min_um": final["cc_min"],
                    "temperature_peak_c": final["temperature_peak_c"],
                    **_area_columns(initial_area[env_id]["summary"], "before"),
                    **_area_columns(final_area["summary"], "final"),
                })
                for tile in final_area["tile_rows"]:
                    tile_rows.append({
                        "surface_kind": args.surface_kind,
                        "direction_mode": args.direction_mode,
                        "surface_profile": profile,
                        "env": env_id,
                        "profile_seed": int(seeds[env_id]),
                        **tile,
                    })
                active.remove(env_id)
            if step % 1000 == 0:
                print(
                    f"[Gate F6] {args.surface_kind}/{args.direction_mode} step={step} "
                    f"complete={env.num_envs - len(active)}/{env.num_envs}", flush=True)

        _write_csv(os.path.join(out_dir, "sequences.csv"), sequence_rows)
        _write_csv(os.path.join(out_dir, "tile_area_fractions.csv"), tile_rows)
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "gate": "F6_ZERO_SHOT",
            "surface_kind": args.surface_kind,
            "curvature_radius_m": float(args.radius_m),
            "freeform_seed": int(args.freeform_seed),
            "direction_mode": args.direction_mode,
            "profiles": PROFILE_IDS,
            "envs_per_profile": int(args.envs_per_profile),
            "surface_seed_base": int(args.surface_seed_base),
            "physics_seed": int(args.physics_seed),
            "policy_seed": int(args.policy_seed),
            "checkpoint": checkpoint,
            "checkpoint_sha256": _sha256(checkpoint),
            "saved_iteration": saved_iteration,
            "observation_dim": obs_dim,
            "action_dim": action_dim,
            "path": dict(env.factory_path_metadata),
            "diagnostic_version": AREA_DIAGNOSTIC_VERSION,
            "completed": len(sequence_rows),
            "expected": env.num_envs,
            "tile_rows": len(tile_rows),
            "training_performed": False,
            "zero_shot": True,
            "all_finite": all(
                np.isfinite(value) for row in sequence_rows for value in row.values()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ),
        }
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        if active:
            raise RuntimeError(f"incomplete F6 run: {len(sequence_rows)}/{env.num_envs}")
        print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
