"""Collect a small paired Gate F7 geometry-aware safety-teacher shard."""
from __future__ import annotations

import argparse
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
parser.add_argument("--surface-kind", choices=("flat", "cylinder", "sphere", "freeform"), required=True)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--envs-per-profile", type=int, default=1)
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
from learning.rl.gate_f7_observation import NORMAL_CURVATURE20, observation_dim  # noqa: E402


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_parent(path: str, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["actor_state_dict"]
    if tuple(state["mlp.0.weight"].shape) != (128, 14):
        raise ValueError("F7 parent must be a 14-D actor")
    dummy = TensorDict({"policy": torch.zeros(1, 14, device=device)}, batch_size=[1])
    actor = MLPModel(
        dummy, {"actor": ["policy"]}, "actor", 2,
        hidden_dims=[128, 128], activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3,
                          "std_type": "scalar"}).to(device)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor


def _teacher(parent_actions: torch.Tensor, observations20: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Ignore normal-z: flat is exactly (0,0,1,0,0,0).  Any appreciable tilt or
    # curvature smoothly activates the F6-derived force-action cap of +0.50.
    signal = torch.cat((observations20[:, 14:16].abs(), observations20[:, 17:20].abs()), dim=1)
    risk = (signal.amax(dim=1) / 0.02).clamp(0.0, 1.0)
    cap = 1.0 - 0.5 * risk
    teacher = parent_actions.clone()
    teacher[:, 0] = torch.minimum(teacher[:, 0], cap)
    return teacher.clamp(-1.0, 1.0), risk


def main() -> None:
    if args.steps <= 0 or args.envs_per_profile <= 0:
        raise ValueError("steps and envs-per-profile must be positive")
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
    cfg.factory_gate_c_direction_mode = "same_xx"
    cfg.surface_kind = args.surface_kind
    cfg.align_pad_to_surface_normal = True
    cfg.gate_f7_observation_mode = NORMAL_CURVATURE20
    cfg.observation_space = observation_dim(NORMAL_CURVATURE20)
    cfg.enable_pad_physical_contact = True
    cfg.repolish_max_passes = 1
    cfg.seed = 20260901
    cfg.surface_seed_base = 32000
    cfg.episode_length_s = 840.0
    env = GateF7CurvedObservationEnv(cfg, render_mode=None)
    env._repolish_mode = True
    parent = _load_parent(checkpoint, str(env.device))
    torch.manual_seed(20260831)
    obs, _ = env.reset()
    observations = []
    parent_actions = []
    teacher_actions = []
    risks = []
    sensor_fault_steps = 0
    force_violation_steps = 0
    try:
        for step in range(args.steps):
            full = obs["policy"]
            if full.shape[1] != 20 or not bool(torch.isfinite(full).all()):
                raise RuntimeError("invalid F7 collection observation")
            with torch.no_grad():
                td = TensorDict({"policy": full[:, :14]}, batch_size=[env.num_envs])
                parent_action = parent(td).clamp(-1.0, 1.0)
                teacher_action, risk = _teacher(parent_action, full)
            observations.append(full.detach().cpu().numpy().astype(np.float32))
            parent_actions.append(parent_action.detach().cpu().numpy().astype(np.float32))
            teacher_actions.append(teacher_action.detach().cpu().numpy().astype(np.float32))
            risks.append(risk.detach().cpu().numpy().astype(np.float32))
            obs, _, _, _, _ = env.step(teacher_action)
            sensor_fault_steps += int(env._sensor_fault.sum())
            force_violation_steps += int(env._force_hard_violated.sum())
            if (step + 1) % 400 == 0:
                print(f"[Gate F7 collect] {args.surface_kind} step={step + 1}/{args.steps}", flush=True)
        x = np.concatenate(observations)
        parent_y = np.concatenate(parent_actions)
        teacher_y = np.concatenate(teacher_actions)
        risk = np.concatenate(risks)
        env_ids = np.tile(np.arange(env.num_envs, dtype=np.int16), args.steps)
        sample_steps = np.repeat(np.arange(args.steps, dtype=np.int32), env.num_envs)
        np.savez_compressed(
            os.path.join(out_dir, "dataset.npz"), observations20=x,
            parent_actions=parent_y, teacher_actions=teacher_y,
            geometry_risk=risk, env_ids=env_ids, control_steps=sample_steps)
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "gate": "F7_SMALL_BC_COLLECTION",
            "surface_kind": args.surface_kind,
            "profiles": PROFILE_IDS,
            "envs_per_profile": args.envs_per_profile,
            "steps": args.steps,
            "samples": len(x),
            "checkpoint": checkpoint,
            "checkpoint_sha256": _sha256(checkpoint),
            "teacher": "parent action with geometry-risk force cap interpolating to +0.50",
            "risk_mean": float(risk.mean()),
            "teacher_changed_fraction": float(np.any(parent_y != teacher_y, axis=1).mean()),
            "sensor_fault_steps": sensor_fault_steps,
            "latched_force_violation_step_sum": force_violation_steps,
            "training_performed": False,
        }
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
        if sensor_fault_steps:
            raise RuntimeError("sensor fault during F7 collection")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
