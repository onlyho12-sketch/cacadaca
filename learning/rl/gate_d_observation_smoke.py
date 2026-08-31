"""Short zero-action PhysX smoke for the full Gate D observation schema."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--profiles", required=True)
parser.add_argument("--envs_per_profile", type=int, default=1)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--out_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from learning.polytwin.factory_surface_profiles import FACTORY_PROFILE_IDS  # noqa: E402
from learning.rl.env.gate_d_factory_observation_env import (  # noqa: E402
    GateDFactoryObservationEnv,
)
from learning.rl.env.gate_d_factory_observation_env_cfg import (  # noqa: E402
    GateDFactoryObservationEnvCfg,
)
from learning.rl.env.gate_d_observation import (  # noqa: E402
    BASE14,
    GLOBAL20,
    SPATIAL120,
    observation_dim,
    observation_feature_names,
)


def _write(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    profiles = tuple(item.strip() for item in args.profiles.split(",") if item.strip())
    if not profiles or len(set(profiles)) != len(profiles):
        raise ValueError("--profiles must contain distinct ids")
    if any(profile not in FACTORY_PROFILE_IDS for profile in profiles):
        raise ValueError(f"profiles must be drawn from {FACTORY_PROFILE_IDS}")
    if args.envs_per_profile <= 0 or args.steps <= 0:
        raise ValueError("envs_per_profile and steps must be positive")
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)

    cfg = GateDFactoryObservationEnvCfg()
    cfg.gate_d_observation_mode = SPATIAL120
    cfg.observation_space = observation_dim(SPATIAL120)
    cfg.factory_profile_ids = profiles
    cfg.factory_envs_per_profile = args.envs_per_profile
    cfg.factory_path_mode = "gate_c"
    cfg.factory_gate_c_step_over_ratio = 0.40
    cfg.factory_gate_c_edge_mode = "balanced_extend5"
    cfg.factory_gate_c_direction_mode = "same_xx"
    cfg.scene.num_envs = len(profiles) * args.envs_per_profile
    cfg.robot_feed_speed_mm_s = 12.7
    cfg.enable_pad_physical_contact = True
    cfg.repolish_max_passes = 1
    cfg.episode_length_s = 840.0
    env = GateDFactoryObservationEnv(cfg, render_mode=None)
    env._repolish_mode = True
    obs, _ = env.reset()

    modes = (BASE14, GLOBAL20, SPATIAL120)
    sample_steps = {0, 1, 5, 10, args.steps}
    sample_rows = []
    seed_rows = []
    for env_id in range(env.num_envs):
        meta = env._factory_metadata[env_id]
        seed_rows.append({
            "env": env_id,
            "surface_profile": meta["profile_id"],
            "profile_seed": int(meta["profile_seed"]),
        })

    def capture(step: int, tensor: torch.Tensor) -> None:
        values = tensor.detach().cpu().numpy()
        if values.shape != (env.num_envs, 120) or not np.isfinite(values).all():
            raise RuntimeError(f"invalid full observation at step {step}: {values.shape}")
        for env_id in range(env.num_envs):
            meta = env._factory_metadata[env_id]
            for mode in modes:
                dim = observation_dim(mode)
                names = observation_feature_names(mode)
                sliced = values[env_id, :dim]
                for index, (name, value) in enumerate(zip(names, sliced)):
                    sample_rows.append({
                        "step": step,
                        "env": env_id,
                        "surface_profile": meta["profile_id"],
                        "profile_seed": int(meta["profile_seed"]),
                        "observation_mode": mode,
                        "observation_dim": dim,
                        "feature_index": index,
                        "feature_name": name,
                        "value": float(value),
                    })
        # Prefix equality is guaranteed by slicing the same state, but assert it
        # explicitly so the smoke records the intended paired-ablation invariant.
        np.testing.assert_array_equal(values[:, :14], values[:, :observation_dim(GLOBAL20)][:, :14])
        np.testing.assert_array_equal(values[:, :20], values[:, :observation_dim(SPATIAL120)][:, :20])

    capture(0, obs["policy"])
    actions = torch.zeros((env.num_envs, 2), device=env.device)
    completed = False
    for step in range(1, args.steps + 1):
        obs, _, terminated, truncated, _ = env.step(actions)
        if bool((terminated | truncated).any()):
            completed = True
            raise RuntimeError("short Gate D smoke unexpectedly completed an episode")
        if step in sample_steps:
            capture(step, obs["policy"])

    summary_rows = []
    for mode in modes:
        for name in observation_feature_names(mode):
            values = np.asarray([
                row["value"] for row in sample_rows
                if row["observation_mode"] == mode and row["feature_name"] == name
            ])
            summary_rows.append({
                "observation_mode": mode,
                "observation_dim": observation_dim(mode),
                "feature_name": name,
                "n": len(values),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            })
    _write(os.path.join(out_dir, "seed_manifest.csv"), seed_rows)
    _write(os.path.join(out_dir, "observation_samples.csv"), sample_rows)
    _write(os.path.join(out_dir, "observation_summary.csv"), summary_rows)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "profiles": profiles,
            "envs_per_profile": args.envs_per_profile,
            "num_envs": env.num_envs,
            "steps": args.steps,
            "sample_steps": sorted(sample_steps),
            "observation_modes": {mode: observation_dim(mode) for mode in modes},
            "same_state_prefix_exact": True,
            "physical_contact": True,
            "zero_action": True,
            "episode_completed": completed,
            "training_performed": False,
            "normalization_status": "PT-DESIGN_NOT_MEASURED_DISTRIBUTION",
            "path": dict(env.factory_path_metadata),
        }, handle, indent=2, sort_keys=True)
    print(
        f"Gate D PhysX smoke: {env.num_envs} envs x {args.steps} steps, "
        "14/20/120 prefixes exact, all finite")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
