"""Short PhysX smoke for the isolated Gate F7 observation environment."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--surface-kind", choices=("flat", "cylinder", "sphere", "freeform"), required=True)
parser.add_argument("--mode", choices=("base14", "normal17", "normal_curvature20"), default="normal_curvature20")
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--out-dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from learning.polytwin.factory_surface_profiles import FACTORY_PREPOLISH  # noqa: E402
from learning.rl.env.gate_f7_curved_observation_env import GateF7CurvedObservationEnv  # noqa: E402
from learning.rl.env.gate_f7_curved_observation_env_cfg import GateF7CurvedObservationEnvCfg  # noqa: E402
from learning.rl.gate_f7_observation import observation_dim  # noqa: E402


def main() -> None:
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)
    cfg = GateF7CurvedObservationEnvCfg()
    cfg.factory_profile_ids = (FACTORY_PREPOLISH,)
    cfg.factory_envs_per_profile = 1
    cfg.scene.num_envs = 1
    cfg.surface_kind = args.surface_kind
    cfg.align_pad_to_surface_normal = True
    cfg.gate_f7_observation_mode = args.mode
    cfg.observation_space = observation_dim(args.mode)
    cfg.enable_pad_physical_contact = True
    cfg.seed = 20260901
    cfg.surface_seed_base = 32000
    env = GateF7CurvedObservationEnv(cfg, render_mode=None)
    finite = True
    sensor_fault_steps = 0
    observed_min = torch.full((observation_dim(args.mode),), float("inf"), device=env.device)
    observed_max = torch.full((observation_dim(args.mode),), float("-inf"), device=env.device)
    try:
        obs, _ = env.reset()
        for _ in range(args.steps):
            values = obs["policy"]
            finite &= bool(torch.isfinite(values).all())
            observed_min = torch.minimum(observed_min, values[0])
            observed_max = torch.maximum(observed_max, values[0])
            obs, _, _, _, _ = env.step(torch.zeros((1, 2), device=env.device))
            sensor_fault_steps += int(env._sensor_fault.sum())
        result = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "gate": "F7_OBSERVATION_PHYSX_SMOKE",
            "surface_kind": args.surface_kind,
            "mode": args.mode,
            "observation_dim": observation_dim(args.mode),
            "steps": args.steps,
            "all_finite": finite,
            "sensor_fault_steps": sensor_fault_steps,
            "observation_min": observed_min.cpu().tolist(),
            "observation_max": observed_max.cpu().tolist(),
            "smoke_pass": finite and sensor_fault_steps == 0,
            "training_performed": False,
        }
        with open(os.path.join(out_dir, "smoke.json"), "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        if not result["smoke_pass"]:
            raise RuntimeError("F7 observation PhysX smoke failed")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
