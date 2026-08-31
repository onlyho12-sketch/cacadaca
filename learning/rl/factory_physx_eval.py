"""Grouped factory-profile PhysX evaluator for corrected Gate B2 and Gate C."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--profiles", required=True,
                    help="comma-separated factory profile ids")
parser.add_argument("--envs_per_profile", type=int, required=True)
parser.add_argument("--path_mode", choices=("gate_b", "gate_c"), default="gate_b")
parser.add_argument("--direction_mode", choices=("same_xx", "cross_xy"), default="same_xx")
parser.add_argument("--step_over_ratio", type=float, default=0.40)
parser.add_argument("--edge_mode", choices=("legacy_start", "balanced", "balanced_extend5"),
                    default="balanced_extend5")
parser.add_argument("--max_passes", type=int, default=1)
parser.add_argument("--max_control_steps", type=int, default=90000)
parser.add_argument("--out_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.models.mlp_model import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from learning.polytwin.factory_surface_profiles import (  # noqa: E402
    FACTORY_PROFILE_IDS,
    FACTORY_SPECS,
)
from learning.rl.env.factory_planar_roi_polish_env import (  # noqa: E402
    FactoryPlanarRoiPolishEnv,
)
from learning.rl.env.factory_planar_roi_polish_env_cfg import (  # noqa: E402
    FactoryPlanarRoiPolishEnvCfg,
)


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_policy(checkpoint: str, device: str):
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ck["actor_state_dict"]
    obs_dim = int(state["mlp.0.weight"].shape[1])
    dummy = TensorDict({"policy": torch.zeros(1, obs_dim, device=device)}, batch_size=[1])
    actor = MLPModel(
        dummy, {"actor": ["policy"]}, "actor", 2,
        hidden_dims=[128, 128], activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3,
                          "std_type": "scalar"}).to(device)
    actor.load_state_dict(state)
    actor.eval()

    def policy(obs):
        td = TensorDict(
            {"policy": obs["policy"][:, :obs_dim]},
            batch_size=[len(obs["policy"])])
        return actor(td).clamp(-1.0, 1.0)
    return policy, obs_dim


def _tile_rows(env_id: int, profile: str, stage: str,
               pass_number: int, maps: dict) -> list[dict]:
    arrays = {name: np.asarray(value, dtype=float) for name, value in maps.items()}
    rows = []
    for tile_x, tile_y in np.ndindex(next(iter(arrays.values())).shape):
        row = {
            "surface_profile": profile,
            "env": env_id,
            "stage": stage,
            "pass": pass_number,
            "tile_x": tile_x,
            "tile_y": tile_y,
        }
        for name, values in arrays.items():
            row[name] = float(values[tile_x, tile_y])
        rows.append(row)
    return rows


def main() -> None:
    profiles = tuple(item.strip() for item in args.profiles.split(",") if item.strip())
    if not profiles or len(set(profiles)) != len(profiles):
        raise ValueError("--profiles must contain distinct profile ids")
    if any(profile not in FACTORY_PROFILE_IDS for profile in profiles):
        raise ValueError(f"factory profiles must be drawn from {FACTORY_PROFILE_IDS}")
    if args.envs_per_profile <= 0 or args.max_passes <= 0:
        raise ValueError("envs_per_profile and max_passes must be positive")

    checkpoint = os.path.abspath(args.checkpoint)
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite existing output directory: {out_dir}")
    os.makedirs(out_dir)

    cfg = FactoryPlanarRoiPolishEnvCfg()
    cfg.factory_profile_ids = profiles
    cfg.factory_envs_per_profile = args.envs_per_profile
    cfg.factory_path_mode = args.path_mode
    cfg.factory_gate_c_step_over_ratio = args.step_over_ratio
    cfg.factory_gate_c_edge_mode = args.edge_mode
    cfg.factory_gate_c_direction_mode = args.direction_mode
    cfg.scene.num_envs = len(profiles) * args.envs_per_profile
    cfg.robot_feed_speed_mm_s = 12.7
    cfg.enable_pad_physical_contact = True
    cfg.repolish_max_passes = args.max_passes
    cfg.repolish_cooldown_s = 20.0
    cfg.episode_length_s = args.max_passes * 720.0 + 120.0
    env = FactoryPlanarRoiPolishEnv(cfg, render_mode=None)
    expected_num_envs = int(env.num_envs)
    env._repolish_mode = True
    policy, obs_dim = _load_policy(checkpoint, env.device)
    obs, _ = env.reset()

    initial_rows, pass_rows, sequence_rows, tile_rows = [], [], [], []
    for env_id in range(env.num_envs):
        diagnostic = env.factory_diagnostic(env_id)
        profile = str(diagnostic["scalars"]["surface_profile"])
        initial_rows.append({"env": env_id, **diagnostic["scalars"]})
        tile_rows.extend(_tile_rows(
            env_id, profile, "initial", 0, diagnostic["tile_maps"]))

    step, completed = 0, set()
    while len(completed) < env.num_envs and step < args.max_control_steps:
        with torch.no_grad():
            actions = policy(obs)
        obs, _, terminated, truncated, _ = env.step(actions)
        step += 1
        done_ids = (terminated | truncated).nonzero(
            as_tuple=False).squeeze(-1).cpu().tolist()
        for env_id in done_ids:
            if env_id in completed:
                continue
            log = env._repolish_log.pop(env_id, None)
            if log is None:
                continue
            initial = initial_rows[env_id]
            profile = str(initial["surface_profile"])
            before, final = log["before"], log["final"]
            sequence_rows.append({
                "surface_profile": profile,
                "path_mode": args.path_mode,
                "direction_mode": env.factory_path_metadata["direction_mode"],
                "checkpoint": os.path.basename(checkpoint),
                "env": env_id,
                "profile_seed": int(initial["profile_seed"]),
                "outcome": log["outcome"],
                "passes": log["passes"],
                "gu_before": before["gu"],
                "gu_final": final["gu"],
                "ra_before_um": before["ra"],
                "ra_final_um": final["ra"],
                "rz_before_um": before["rz"],
                "rz_final_um": final["rz"],
                "scratch_before_um": before["scratch"],
                "scratch_final_um": final["scratch"],
                "clearcoat_min_um": final["cc_min"],
                "quality_ok": log["quality_ok"],
                "safety_ok": log["safety_ok"],
            })
            for item in log.get("pass_history", []):
                row = {
                    "surface_profile": profile,
                    "path_mode": args.path_mode,
                    "direction_mode": env.factory_path_metadata["direction_mode"],
                    "checkpoint": os.path.basename(checkpoint),
                    "env": env_id,
                    "profile_seed": int(initial["profile_seed"]),
                    **item,
                }
                pass_rows.append(row)
                tile_rows.extend(_tile_rows(
                    env_id, profile, "after_pass", int(item["pass"]),
                    json.loads(item["factory_tile_maps_json"])))
            completed.add(env_id)
        if step % 1000 == 0:
            print(
                f"[Factory eval] step={step} complete={len(completed)}/{env.num_envs}",
                flush=True)

    _write_csv(os.path.join(out_dir, "initial_diagnostics.csv"), initial_rows)
    _write_csv(os.path.join(out_dir, "sequences.csv"), sequence_rows)
    _write_csv(os.path.join(out_dir, "passes.csv"), pass_rows)
    _write_csv(os.path.join(out_dir, "tiles.csv"), tile_rows)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profiles": profiles,
        "envs_per_profile": args.envs_per_profile,
        "num_envs": env.num_envs,
        "max_passes": args.max_passes,
        "physical_contact": True,
        "feed_speed_mm_s": 12.7,
        "steps_executed": step,
        "complete": len(completed) == env.num_envs,
        "profile_seeds": {
            profile: [int(row["profile_seed"]) for row in initial_rows
                      if row["surface_profile"] == profile]
            for profile in profiles
        },
        "path": dict(env.factory_path_metadata),
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(checkpoint),
        "obs_dim": obs_dim,
        "factory_specs": {profile: asdict(FACTORY_SPECS[profile]) for profile in profiles},
        "surface_state_extended": False,
        "mar_density_or_severity_created": False,
        "training_performed": False,
        "geometry": env.roi_geometry.validate(),
        "caveat": "All factory profile ranges are PT-DESIGN, not measured distributions.",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(f"[Factory eval] wrote {out_dir}")
    print(f"[Factory eval] complete={len(completed)}/{expected_num_envs} steps={step}")
    env.close()
    if len(completed) != expected_num_envs:
        raise RuntimeError(f"incomplete run: {len(completed)}/{expected_num_envs}")


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
