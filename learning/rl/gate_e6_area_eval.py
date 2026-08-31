"""Gate E6 paired evaluation with sub-tile four-target area accounting."""
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

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--policy_label", required=True)
parser.add_argument("--direction_mode", choices=("same_xx", "cross_xy"), required=True)
parser.add_argument("--envs_per_profile", type=int, default=2)
parser.add_argument("--surface_seed_base", type=int, default=18500)
parser.add_argument("--max_control_steps", type=int, default=120000)
parser.add_argument("--out_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.models.mlp_model import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from learning.polytwin.factory_surface_profiles import PROFILE_IDS  # noqa: E402
from learning.rl.env.gate_d_observation import SPATIAL120, observation_dim  # noqa: E402
from learning.rl.env.gate_e_mixed_observation_env import GateEMixedObservationEnv  # noqa: E402
from learning.rl.env.gate_e_mixed_observation_env_cfg import GateEMixedObservationEnvCfg  # noqa: E402
from learning.rl.env.planar_roi_diagnostics import surface_view  # noqa: E402
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
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_policy(checkpoint: str, device: str):
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ck["actor_state_dict"]
    obs_dim = int(state["mlp.0.weight"].shape[1])
    if obs_dim not in (14, 20, 120):
        raise ValueError(f"unsupported Gate E observation dimension: {obs_dim}")
    dummy = TensorDict({"policy": torch.zeros(1, obs_dim, device=device)}, batch_size=[1])
    actor = MLPModel(
        dummy, {"actor": ["policy"]}, "actor", 2,
        hidden_dims=[128, 128], activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3,
                          "std_type": "scalar"}).to(device)
    actor.load_state_dict(state)
    actor.eval()

    def policy(observation120: torch.Tensor) -> torch.Tensor:
        td = TensorDict(
            {"policy": observation120[:, :obs_dim]}, batch_size=[len(observation120)])
        return actor(td).clamp(-1.0, 1.0)

    return policy, obs_dim, ck.get("infos", {})


class GateE6AreaEnv(GateEMixedObservationEnv):
    """Capture final high-resolution maps immediately before framework reset."""

    def __init__(self, *args, **kwargs):
        self.gate_e6_final_area: dict[int, dict] = {}
        super().__init__(*args, **kwargs)

    def _capture_area(self, env_id: int) -> None:
        roi = surface_view(self._surfaces[env_id], self._roi_slices)
        targets = AreaQualityTargets(
            ra_max_um=float(self.cfg.t_ra_pass_max_um),
            rz_max_um=float(self.cfg.t_rz_pass_max_um),
            clearcoat_min_um=float(self.cfg.clearcoat_safety_limit_um),
            temperature_max_c=float(self.cfg.thermal_hard_limit_c),
        )
        self.gate_e6_final_area[env_id] = summarize_area_quality(roi, targets)

    def _repolish_decide(self, env_id: int) -> bool:
        terminate = super()._repolish_decide(env_id)
        if terminate:
            self._capture_area(env_id)
        return terminate

    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        done_ids = (terminated | truncated).nonzero(
            as_tuple=False).squeeze(-1).cpu().tolist()
        for env_id in done_ids:
            if env_id not in self.gate_e6_final_area:
                self._capture_area(env_id)
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
    if args.envs_per_profile <= 0:
        raise ValueError("envs_per_profile must be positive")
    checkpoint = os.path.abspath(args.checkpoint)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)

    cfg = GateEMixedObservationEnvCfg()
    cfg.factory_profile_ids = PROFILE_IDS
    cfg.factory_envs_per_profile = args.envs_per_profile
    cfg.factory_path_mode = "gate_c"
    cfg.factory_gate_c_step_over_ratio = 0.40
    cfg.factory_gate_c_edge_mode = "balanced_extend5"
    cfg.factory_gate_c_direction_mode = args.direction_mode
    cfg.gate_d_observation_mode = SPATIAL120
    cfg.observation_space = observation_dim(SPATIAL120)
    cfg.scene.num_envs = len(PROFILE_IDS) * args.envs_per_profile
    cfg.seed = args.surface_seed_base
    cfg.surface_seed_base = args.surface_seed_base
    cfg.robot_feed_speed_mm_s = 12.7
    cfg.enable_pad_physical_contact = True
    cfg.repolish_max_passes = 1
    cfg.repolish_cooldown_s = 20.0
    cfg.episode_length_s = 840.0
    env = GateE6AreaEnv(cfg, render_mode=None)
    env._repolish_mode = True
    policy, obs_dim, checkpoint_info = _load_policy(checkpoint, str(env.device))
    obs, _ = env.reset()
    seeds = np.asarray([
        int(env._factory_metadata[env_id]["profile_seed"])
        for env_id in range(env.num_envs)
    ], dtype=np.int64)
    initial_area = {}
    for env_id in range(env.num_envs):
        roi = surface_view(env._surfaces[env_id], env._roi_slices)
        targets = AreaQualityTargets(
            ra_max_um=float(cfg.t_ra_pass_max_um),
            rz_max_um=float(cfg.t_rz_pass_max_um),
            clearcoat_min_um=float(cfg.clearcoat_safety_limit_um),
            temperature_max_c=float(cfg.thermal_hard_limit_c),
        )
        initial_area[env_id] = summarize_area_quality(roi, targets)

    active = set(range(env.num_envs))
    action_sum = np.zeros((env.num_envs, 2), dtype=np.float64)
    control_steps = np.zeros(env.num_envs, dtype=np.int64)
    contact_steps = np.zeros(env.num_envs, dtype=np.int64)
    sensor_fault_steps = np.zeros(env.num_envs, dtype=np.int64)
    sequence_rows: list[dict] = []
    tile_rows: list[dict] = []
    step = 0
    while active and step < args.max_control_steps:
        full = obs["policy"]
        if not bool(torch.isfinite(full).all()):
            raise RuntimeError("Gate E6 evaluation observation contains NaN/Inf")
        with torch.no_grad():
            actions = policy(full)
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
        for env_id in active:
            contact_steps[env_id] += int(used[env_id] > 0.05)
            sensor_fault_steps[env_id] += int(faults[env_id])
        done_ids = (terminated | truncated).nonzero(
            as_tuple=False).squeeze(-1).cpu().tolist()
        for env_id in done_ids:
            if env_id not in active:
                continue
            log = env._repolish_log.pop(env_id, None)
            final_area = env.gate_e6_final_area.pop(env_id, None)
            if log is None or final_area is None:
                raise RuntimeError(f"missing final diagnostic for env {env_id}")
            profile = str(env._factory_metadata[env_id]["profile_id"])
            before, final = log["before"], log["final"]
            n = max(1, int(control_steps[env_id]))
            row = {
                "policy_label": args.policy_label,
                "observation_dim": obs_dim,
                "direction_mode": args.direction_mode,
                "surface_profile": profile,
                "env": env_id,
                "profile_seed": int(seeds[env_id]),
                "outcome": log["outcome"],
                "quality_ok": log["quality_ok"],
                "safety_ok": log["safety_ok"],
                "control_steps": n,
                "contact_steps": int(contact_steps[env_id]),
                "sensor_fault_steps": int(sensor_fault_steps[env_id]),
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
            }
            sequence_rows.append(row)
            for tile in final_area["tile_rows"]:
                tile_rows.append({
                    "policy_label": args.policy_label,
                    "observation_dim": obs_dim,
                    "direction_mode": args.direction_mode,
                    "surface_profile": profile,
                    "env": env_id,
                    "profile_seed": int(seeds[env_id]),
                    **tile,
                })
            active.remove(env_id)
        if step % 1000 == 0:
            print(
                f"[Gate E6 area eval] {args.policy_label}/{args.direction_mode} "
                f"step={step} complete={env.num_envs - len(active)}/{env.num_envs}",
                flush=True)

    _write_csv(os.path.join(out_dir, "sequences.csv"), sequence_rows)
    _write_csv(os.path.join(out_dir, "tile_area_fractions.csv"), tile_rows)
    metadata_out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "Gate E6 paired evaluation with sub-tile quality area",
        "diagnostic_version": AREA_DIAGNOSTIC_VERSION,
        "design_status": "PT-DESIGN local-window extension; SYNTHETIC output",
        "episode_decision_changed": False,
        "policy_label": args.policy_label,
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_infos": checkpoint_info,
        "observation_dim": obs_dim,
        "profiles": PROFILE_IDS,
        "envs_per_profile": args.envs_per_profile,
        "surface_seed_base": args.surface_seed_base,
        "profile_seeds": seeds[:args.envs_per_profile].tolist(),
        "path": dict(env.factory_path_metadata),
        "completed": len(sequence_rows),
        "expected": env.num_envs,
        "tile_rows": len(tile_rows),
        "quality_four": [
            "local GU >= 70", "local Ra <= 0.20 um", "local Rz <= 2.0 um",
            "scratch improved from initial or initial < 0.05 um",
        ],
        "safety_separate": ["clearcoat", "temperature", "force/contact"],
        "training_performed_in_eval": False,
        "champion_promoted": False,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata_out, handle, indent=2, sort_keys=True)
    env.close()
    print(
        f"[Gate E6 area eval] wrote {out_dir}; "
        f"complete={len(sequence_rows)}/{cfg.scene.num_envs}", flush=True)
    if active:
        raise RuntimeError(f"incomplete Gate E6 evaluation: {len(sequence_rows)}/{env.num_envs}")


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()

