"""Gate B2 minimal physical-contact diagnostic for profiled planar ROI surfaces."""
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
parser.add_argument("--surface_profile", choices=("legacy_stress", "new_car_mild"),
                    default="new_car_mild")
parser.add_argument("--num_envs", type=int, default=4)
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

from learning.polytwin.mar_optics import DEFAULT_MAR_OPTICS  # noqa: E402
from learning.polytwin.surface_profiles import NEW_CAR_MILD_SPEC  # noqa: E402
from learning.rl.env.profiled_planar_roi_polish_env import (  # noqa: E402
    ProfiledPlanarRoiPolishEnv,
)
from learning.rl.env.profiled_planar_roi_polish_env_cfg import (  # noqa: E402
    ProfiledPlanarRoiPolishEnvCfg,
)


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


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
    actor.load_state_dict(state); actor.eval()

    def policy(obs):
        td = TensorDict({"policy": obs["policy"][:, :obs_dim]}, batch_size=[len(obs["policy"])])
        return actor(td).clamp(-1.0, 1.0)
    return policy, obs_dim


def _tile_rows(env_id: int, stage: str, pass_number: int, maps: dict) -> list[dict]:
    arrays = {name: np.asarray(value, dtype=float) for name, value in maps.items()}
    rows = []
    for tile_x, tile_y in np.ndindex(next(iter(arrays.values())).shape):
        row = {"env": env_id, "stage": stage, "pass": pass_number,
               "tile_x": tile_x, "tile_y": tile_y}
        for name, values in arrays.items():
            row[name] = float(values[tile_x, tile_y])
        rows.append(row)
    return rows


def main() -> None:
    checkpoint = os.path.abspath(args.checkpoint)
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite existing output directory: {out_dir}")
    os.makedirs(out_dir)

    cfg = ProfiledPlanarRoiPolishEnvCfg()
    cfg.surface_profile = args.surface_profile
    cfg.scene.num_envs = args.num_envs
    cfg.robot_feed_speed_mm_s = 12.7
    cfg.enable_pad_physical_contact = True
    cfg.repolish_max_passes = args.max_passes
    cfg.repolish_cooldown_s = 20.0
    cfg.episode_length_s = args.max_passes * 720.0 + 120.0
    env = ProfiledPlanarRoiPolishEnv(cfg, render_mode=None)
    env._repolish_mode = True
    policy, obs_dim = _load_policy(checkpoint, env.device)
    obs, _ = env.reset()

    initial_rows, pass_rows, sequence_rows, tile_rows = [], [], [], []
    for env_id in range(args.num_envs):
        diagnostic = env.profile_diagnostic(env_id)
        initial_rows.append({"env": env_id, **diagnostic["scalars"]})
        tile_rows.extend(_tile_rows(
            env_id, "initial", 0, diagnostic["tile_maps"]))

    step = 0
    completed = set()
    while len(completed) < args.num_envs and step < args.max_control_steps:
        with torch.no_grad():
            actions = policy(obs)
        obs, _, terminated, truncated, _ = env.step(actions)
        step += 1
        done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
        for env_id in done_ids:
            if env_id in completed:
                continue
            log = env._repolish_log.pop(env_id, None)
            if log is None:
                continue
            before, final = log["before"], log["final"]
            sequence_rows.append({
                "surface_profile": args.surface_profile,
                "checkpoint": os.path.basename(checkpoint), "env": env_id,
                "outcome": log["outcome"], "passes": log["passes"],
                "gu_before": before["gu"], "gu_final": final["gu"],
                "ra_before_um": before["ra"], "ra_final_um": final["ra"],
                "rz_before_um": before["rz"], "rz_final_um": final["rz"],
                "scratch_before_um": before["scratch"],
                "scratch_final_um": final["scratch"],
                "clearcoat_min_um": final["cc_min"],
                "quality_ok": log["quality_ok"], "safety_ok": log["safety_ok"],
            })
            for item in log.get("pass_history", []):
                row = {"surface_profile": args.surface_profile,
                       "checkpoint": os.path.basename(checkpoint), "env": env_id, **item}
                pass_rows.append(row)
                tile_rows.extend(_tile_rows(
                    env_id, "after_pass", int(item["pass"]),
                    json.loads(item["profile_tile_maps_json"])))
            completed.add(env_id)
        if step % 2000 == 0:
            print(f"[Gate B2] step={step} complete={len(completed)}/{args.num_envs}", flush=True)

    _write_csv(os.path.join(out_dir, "initial_diagnostics.csv"), initial_rows)
    _write_csv(os.path.join(out_dir, "profile_sequences.csv"), sequence_rows)
    _write_csv(os.path.join(out_dir, "profile_passes.csv"), pass_rows)
    _write_csv(os.path.join(out_dir, "profile_tiles.csv"), tile_rows)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "surface_profile": args.surface_profile, "num_envs": args.num_envs,
        "max_passes": args.max_passes, "physical_contact": True,
        "feed_speed_mm_s": 12.7, "steps_executed": step,
        "complete": len(completed) == args.num_envs,
        "checkpoint": checkpoint, "checkpoint_sha256": _sha256(checkpoint),
        "obs_dim": obs_dim, "new_car_mild_spec": asdict(NEW_CAR_MILD_SPEC),
        "mar_optics_config": asdict(DEFAULT_MAR_OPTICS),
        "geometry": env.roi_geometry.validate(),
        "caveat": "All new_car_mild joint defect and mar distributions are PT-DESIGN.",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(f"[Gate B2] wrote {out_dir}")
    print(f"[Gate B2] complete={len(completed)}/{args.num_envs} steps={step}")
    env.close()
    if len(completed) != args.num_envs:
        raise RuntimeError(f"incomplete Gate B2 run: {len(completed)}/{args.num_envs}")


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
