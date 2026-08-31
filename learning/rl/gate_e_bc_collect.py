"""Collect one successful factory sequence per environment for Gate E2 BC.

Each invocation handles one profile/path stratum and writes an immutable NPZ
shard plus sequence CSV.  Failed or incomplete sequences remain in the audit
CSV but are excluded from supervised labels.  No training is performed.
"""
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
parser.add_argument("--profile", choices=(
    "factory_prepolish",
    "factory_prepolish_deep_defect",
    "factory_prepolish_deep_stress",
), required=True)
parser.add_argument("--direction_mode", choices=("same_xx", "cross_xy"), required=True)
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--surface_seed_base", type=int, default=4200)
parser.add_argument("--max_control_steps", type=int, default=20000)
parser.add_argument("--out_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.models.mlp_model import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from learning.rl.env.gate_d_observation import SPATIAL120, observation_dim  # noqa: E402
from learning.rl.env.gate_e_mixed_observation_env import (  # noqa: E402
    GateEMixedObservationEnv,
)
from learning.rl.env.gate_e_mixed_observation_env_cfg import (  # noqa: E402
    GateEMixedObservationEnvCfg,
)
from learning.rl.gate_e_dataset import (  # noqa: E402
    GateEShard,
    save_shard,
    split_for_seed,
    split_id_for_seed,
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _load_champion(checkpoint: str, device: str):
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ck["actor_state_dict"]
    obs_dim = int(state["mlp.0.weight"].shape[1])
    if obs_dim != 14:
        raise ValueError(f"expected frozen 14-D champion, got {obs_dim}")
    dummy = TensorDict({"policy": torch.zeros(1, obs_dim, device=device)}, batch_size=[1])
    actor = MLPModel(
        dummy, {"actor": ["policy"]}, "actor", 2,
        hidden_dims=[128, 128], activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3,
                          "std_type": "scalar"}).to(device)
    actor.load_state_dict(state); actor.eval()

    def policy(observation120: torch.Tensor) -> torch.Tensor:
        td = TensorDict({"policy": observation120[:, :obs_dim]}, batch_size=[len(observation120)])
        return actor(td).clamp(-1.0, 1.0)
    return policy


def main() -> None:
    if args.num_envs <= 0 or args.max_control_steps <= 0:
        raise ValueError("num_envs and max_control_steps must be positive")
    checkpoint = os.path.abspath(args.checkpoint)
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)

    cfg = GateEMixedObservationEnvCfg()
    cfg.factory_profile_ids = (args.profile,)
    cfg.factory_envs_per_profile = args.num_envs
    cfg.factory_path_mode = "gate_c"
    cfg.factory_gate_c_step_over_ratio = 0.40
    cfg.factory_gate_c_edge_mode = "balanced_extend5"
    cfg.factory_gate_c_direction_mode = args.direction_mode
    cfg.gate_d_observation_mode = SPATIAL120
    cfg.observation_space = observation_dim(SPATIAL120)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.surface_seed_base
    cfg.surface_seed_base = args.surface_seed_base
    cfg.robot_feed_speed_mm_s = 12.7
    cfg.enable_pad_physical_contact = True
    cfg.repolish_max_passes = 1
    cfg.repolish_cooldown_s = 20.0
    cfg.episode_length_s = 840.0
    env = GateEMixedObservationEnv(cfg, render_mode=None)
    env._repolish_mode = True
    policy = _load_champion(checkpoint, env.device)
    obs, _ = env.reset()

    seeds = np.asarray([
        int(env._factory_metadata[env_id]["profile_seed"])
        for env_id in range(args.num_envs)
    ], dtype=np.int64)
    active = set(range(args.num_envs))
    successful: set[int] = set()
    rows: list[dict] = []
    obs_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    env_chunks: list[np.ndarray] = []
    seed_chunks: list[np.ndarray] = []
    contact_steps = np.zeros(args.num_envs, dtype=np.int64)
    sensor_fault_steps = np.zeros(args.num_envs, dtype=np.int64)
    step = 0
    while active and step < args.max_control_steps:
        full = obs["policy"]
        if full.shape != (args.num_envs, observation_dim(SPATIAL120)):
            raise RuntimeError(f"unexpected observation shape {tuple(full.shape)}")
        if not bool(torch.isfinite(full).all()):
            raise RuntimeError("Gate E collection observation contains NaN/Inf")
        with torch.no_grad():
            actions = policy(full)
        ids = np.asarray(sorted(active), dtype=np.int32)
        ids_t = torch.as_tensor(ids, device=env.device, dtype=torch.long)
        obs_chunks.append(full[ids_t].detach().cpu().numpy().astype(np.float32, copy=False))
        action_chunks.append(actions[ids_t].detach().cpu().numpy().astype(np.float32, copy=False))
        env_chunks.append(ids)
        seed_chunks.append(seeds[ids])
        inactive = sorted(set(range(args.num_envs)) - active)
        if inactive:
            actions[inactive] = 0.0
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
            if log is None:
                continue
            ok = bool(log["outcome"] == "success" and log["quality_ok"] and log["safety_ok"])
            if ok:
                successful.add(env_id)
            rows.append({
                "surface_profile": args.profile,
                "direction_mode": args.direction_mode,
                "env": env_id,
                "profile_seed": int(seeds[env_id]),
                "split": split_for_seed(int(seeds[env_id])),
                "outcome": log["outcome"],
                "passes": log["passes"],
                "quality_ok": log["quality_ok"],
                "safety_ok": log["safety_ok"],
                "included_in_bc": ok,
                "contact_steps": int(contact_steps[env_id]),
                "sensor_fault_steps": int(sensor_fault_steps[env_id]),
                "gu_final": log["final"]["gu"],
                "ra_final_um": log["final"]["ra"],
                "rz_final_um": log["final"]["rz"],
                "scratch_final_um": log["final"]["scratch"],
                "clearcoat_min_um": log["final"]["cc_min"],
            })
            active.remove(env_id)
        if step % 1000 == 0:
            print(
                f"[Gate E2 collect] {args.profile}/{args.direction_mode} "
                f"step={step} complete={args.num_envs - len(active)}/{args.num_envs}",
                flush=True)

    all_obs = np.concatenate(obs_chunks)
    all_actions = np.concatenate(action_chunks)
    all_envs = np.concatenate(env_chunks)
    all_seeds = np.concatenate(seed_chunks)
    include = np.isin(all_envs, np.asarray(sorted(successful), dtype=np.int32))
    kept_obs = all_obs[include]
    kept_actions = all_actions[include]
    kept_envs = all_envs[include]
    kept_seeds = all_seeds[include]
    split_ids = np.asarray([split_id_for_seed(seed) for seed in kept_seeds], dtype=np.uint8)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "Gate E2 pilot BC collection",
        "surface_profile": args.profile,
        "direction_mode": args.direction_mode,
        "num_envs": args.num_envs,
        "surface_seed_base": args.surface_seed_base,
        "profile_seeds": seeds.tolist(),
        "split_rule": "SHA256(profile_seed), 70/15/15; shared across profile/path",
        "path": dict(env.factory_path_metadata),
        "teacher": "frozen_champion_pass1",
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(checkpoint),
        "completed_sequences": len(rows),
        "successful_sequences": len(successful),
        "failed_or_incomplete_sequences": args.num_envs - len(successful),
        "samples_before_success_filter": int(len(all_obs)),
        "samples_saved": int(len(kept_obs)),
        "sample_split_counts": {
            name: int(np.sum(split_ids == index))
            for index, name in enumerate(("train", "validation", "test"))
        },
        "physical_contact": True,
        "training_performed": False,
        "execution_noise": 0.0,
        "profile_mix_status": "PT-DESIGN_NOT_MEASURED_PRODUCTION_MIX",
    }
    shard = GateEShard(
        observations120=kept_obs,
        actions=kept_actions,
        profile_seeds=kept_seeds,
        sequence_ids=kept_envs,
        split_ids=split_ids,
    )
    save_shard(os.path.join(out_dir, "dataset.npz"), shard, metadata)
    _write_csv(os.path.join(out_dir, "sequences.csv"), rows)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    env.close()
    print(
        f"[Gate E2 collect] wrote {out_dir}; samples={len(kept_obs)} "
        f"success={len(successful)}/{args.num_envs}")
    if active:
        raise RuntimeError(f"incomplete collection: {args.num_envs - len(active)}/{args.num_envs}")
    if len(successful) != args.num_envs:
        raise RuntimeError(
            f"not all sequences produced valid BC labels: {len(successful)}/{args.num_envs}")


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
