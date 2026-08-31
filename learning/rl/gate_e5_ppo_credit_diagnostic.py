"""Gate E5 frozen-policy PPO credit-assignment diagnostic.

The environment reward is not modified.  An isolated subclass observes the
already computed total reward and the established terminal reward, defining
the dense residual as total minus terminal.  The frozen champion then runs one
complete physical pass on paired factory profiles.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--envs_per_profile", type=int, default=4)
parser.add_argument("--surface_seed_base", type=int, default=17000)
parser.add_argument("--rollout_steps", type=int, default=48)
parser.add_argument("--gamma", type=float, default=0.9995)
parser.add_argument("--comparison_gamma", type=float, default=0.99)
parser.add_argument("--max_control_steps", type=int, default=120000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.models.mlp_model import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from learning.polytwin.factory_surface_profiles import FACTORY_PROFILE_IDS  # noqa: E402
from learning.rl.env.gate_d_observation import BASE14, observation_dim  # noqa: E402
from learning.rl.env.gate_e_mixed_observation_env import GateEMixedObservationEnv  # noqa: E402
from learning.rl.env.gate_e_mixed_observation_env_cfg import (  # noqa: E402
    GateEMixedObservationEnvCfg,
)


class GateE5RewardDiagnosticEnv(GateEMixedObservationEnv):
    """Observe established rewards without changing their returned values."""

    def _get_rewards(self) -> torch.Tensor:
        total = super()._get_rewards()
        terminal = torch.zeros_like(total)
        done_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
        for env_id in done_ids:
            result = self.last_episode_results.get(env_id)
            if result is not None:
                terminal[env_id] = float(result["terminal_reward"])
        self.gate_e5_total_reward = total.detach().clone()
        self.gate_e5_terminal_reward = terminal.detach().clone()
        self.gate_e5_dense_reward = (total - terminal).detach().clone()
        return total


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_policy(checkpoint: str, device: str):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    state = saved["actor_state_dict"]
    dim = int(state["mlp.0.weight"].shape[1])
    if dim != observation_dim(BASE14):
        raise ValueError(f"Gate E5 requires base14 checkpoint, got {dim}")
    dummy = TensorDict({"policy": torch.zeros(1, dim, device=device)}, batch_size=[1])
    actor = MLPModel(
        dummy, {"actor": ["policy"]}, "actor", 2,
        hidden_dims=[128, 128], activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3,
                          "std_type": "scalar"}).to(device)
    actor.load_state_dict(state); actor.eval()

    def policy(observation: torch.Tensor) -> torch.Tensor:
        td = TensorDict({"policy": observation}, batch_size=[len(observation)])
        return actor(td).clamp(-1.0, 1.0)
    return policy, saved.get("infos", {})


def main() -> None:
    if args.envs_per_profile <= 0 or args.rollout_steps <= 0:
        raise ValueError("envs_per_profile and rollout_steps must be positive")
    if not (0.0 < args.gamma <= 1.0 and 0.0 < args.comparison_gamma <= 1.0):
        raise ValueError("gamma values must be in (0, 1]")
    checkpoint = os.path.abspath(args.checkpoint)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)
    checkpoint_hash = sha256(checkpoint)

    cfg = GateEMixedObservationEnvCfg()
    cfg.factory_profile_ids = FACTORY_PROFILE_IDS
    cfg.factory_envs_per_profile = args.envs_per_profile
    cfg.factory_path_mode = "gate_c"
    cfg.factory_gate_c_step_over_ratio = 0.40
    cfg.factory_gate_c_edge_mode = "balanced_extend5"
    cfg.factory_gate_c_direction_mode = "same_xx"
    cfg.gate_d_observation_mode = BASE14
    cfg.observation_space = observation_dim(BASE14)
    cfg.scene.num_envs = len(FACTORY_PROFILE_IDS) * args.envs_per_profile
    cfg.seed = args.surface_seed_base
    cfg.surface_seed_base = args.surface_seed_base
    cfg.robot_feed_speed_mm_s = 12.7
    cfg.enable_pad_physical_contact = True
    cfg.repolish_max_passes = 1
    cfg.repolish_cooldown_s = 20.0
    cfg.episode_length_s = 840.0

    env = None
    try:
        env = GateE5RewardDiagnosticEnv(cfg, render_mode=None)
        env._repolish_mode = True
        policy, checkpoint_infos = load_policy(checkpoint, str(env.device))
        observation, _ = env.reset()
        profile_ids = [
            str(env._factory_metadata[i]["profile_id"]) for i in range(env.num_envs)]
        profile_seeds = [
            int(env._factory_metadata[i]["profile_seed"]) for i in range(env.num_envs)]

        active = set(range(env.num_envs))
        control_steps = np.zeros(env.num_envs, dtype=np.int64)
        contact_steps = np.zeros(env.num_envs, dtype=np.int64)
        fault_steps = np.zeros(env.num_envs, dtype=np.int64)
        total_sum = np.zeros(env.num_envs, dtype=np.float64)
        dense_sum = np.zeros(env.num_envs, dtype=np.float64)
        dense_abs_sum = np.zeros(env.num_envs, dtype=np.float64)
        terminal_sum = np.zeros(env.num_envs, dtype=np.float64)
        terminal_events = np.zeros(env.num_envs, dtype=np.int64)
        trace_rows = []
        sequence_rows = []
        step = 0
        while active and step < args.max_control_steps:
            full = observation["policy"]
            if not bool(torch.isfinite(full).all()):
                raise RuntimeError("Gate E5 observation contains NaN/Inf")
            with torch.no_grad():
                actions = policy(full)
            inactive = sorted(set(range(env.num_envs)) - active)
            if inactive:
                actions[inactive] = 0.0
            observation, rewards, terminated, truncated, _ = env.step(actions)
            step += 1
            total = env.gate_e5_total_reward.detach().cpu().numpy()
            dense = env.gate_e5_dense_reward.detach().cpu().numpy()
            terminal_reward = env.gate_e5_terminal_reward.detach().cpu().numpy()
            if not (np.isfinite(total).all() and np.isfinite(dense).all()
                    and np.isfinite(terminal_reward).all()):
                raise RuntimeError("Gate E5 reward contains NaN/Inf")
            active_before = sorted(active)
            used = env._force_used_n.detach().cpu().numpy()
            faults = env._sensor_fault.detach().cpu().numpy()
            for env_id in active_before:
                control_steps[env_id] += 1
                contact_steps[env_id] += int(used[env_id] > 0.05)
                fault_steps[env_id] += int(faults[env_id])
                total_sum[env_id] += float(total[env_id])
                dense_sum[env_id] += float(dense[env_id])
                dense_abs_sum[env_id] += abs(float(dense[env_id]))
                terminal_sum[env_id] += float(terminal_reward[env_id])
                terminal_events[env_id] += int(abs(float(terminal_reward[env_id])) > 0.0)
            trace_rows.append({
                "control_step": step,
                "rollout_iteration_zero_based": (step - 1) // args.rollout_steps,
                "active_envs": len(active_before),
                "reward_mean_active": float(np.mean(total[active_before])),
                "reward_min_active": float(np.min(total[active_before])),
                "reward_max_active": float(np.max(total[active_before])),
                "dense_mean_active": float(np.mean(dense[active_before])),
                "dense_abs_mean_active": float(np.mean(np.abs(dense[active_before]))),
                "terminal_sum_active": float(np.sum(terminal_reward[active_before])),
                "terminal_nonzero_events": int(np.count_nonzero(terminal_reward[active_before])),
            })

            done_ids = (terminated | truncated).nonzero(
                as_tuple=False).squeeze(-1).cpu().tolist()
            for env_id in done_ids:
                if env_id not in active:
                    continue
                log = env._repolish_log.pop(env_id, None)
                if log is None:
                    raise RuntimeError(f"missing Gate E5 completion log for env {env_id}")
                n = int(control_steps[env_id])
                direct_steps = min(args.rollout_steps, n)
                uninterrupted_weight = args.gamma ** max(0, n - 1)
                comparison_weight = args.comparison_gamma ** max(0, n - 1)
                before, final = log["before"], log["final"]
                sequence_rows.append({
                    "env": env_id,
                    "surface_profile": profile_ids[env_id],
                    "profile_seed": profile_seeds[env_id],
                    "outcome": log["outcome"],
                    "quality_ok": log["quality_ok"],
                    "safety_ok": log["safety_ok"],
                    "control_steps": n,
                    "rollout_iteration_of_terminal_zero_based": (n - 1) // args.rollout_steps,
                    "rollout_iterations_required": math.ceil(n / args.rollout_steps),
                    "direct_terminal_credit_steps_max": direct_steps,
                    "direct_terminal_credit_path_fraction": direct_steps / n,
                    "contact_steps": int(contact_steps[env_id]),
                    "sensor_fault_steps": int(fault_steps[env_id]),
                    "total_reward_sum": float(total_sum[env_id]),
                    "dense_reward_sum": float(dense_sum[env_id]),
                    "dense_reward_abs_sum": float(dense_abs_sum[env_id]),
                    "terminal_reward_sum": float(terminal_sum[env_id]),
                    "terminal_reward_events": int(terminal_events[env_id]),
                    "terminal_to_dense_abs_ratio": (
                        abs(float(terminal_sum[env_id]))
                        / max(float(dense_abs_sum[env_id]), 1e-12)),
                    "hypothetical_uninterrupted_gamma_weight": uninterrupted_weight,
                    "hypothetical_uninterrupted_terminal_at_start": (
                        uninterrupted_weight * float(terminal_sum[env_id])),
                    "comparison_gamma_weight": comparison_weight,
                    "gu_before": before["gu"], "gu_final": final["gu"],
                    "ra_before_um": before["ra"], "ra_final_um": final["ra"],
                    "rz_before_um": before["rz"], "rz_final_um": final["rz"],
                    "scratch_before_um": before["scratch"],
                    "scratch_final_um": final["scratch"],
                    "clearcoat_min_um": final["cc_min"],
                    "temperature_peak_c": final["temperature_peak_c"],
                })
                active.remove(env_id)
            if step % 1000 == 0:
                print(
                    f"[Gate E5] step={step} complete={env.num_envs - len(active)}/"
                    f"{env.num_envs}", flush=True)

        if active:
            raise RuntimeError(f"incomplete Gate E5 diagnostic: {len(active)} active")
        sequence_rows.sort(key=lambda row: int(row["env"]))
        max_steps = max(int(row["control_steps"]) for row in sequence_rows)
        coverage_rows = []
        cumulative = 0
        for iteration in range(math.ceil(max_steps / args.rollout_steps)):
            start = iteration * args.rollout_steps
            end = (iteration + 1) * args.rollout_steps
            events = sum(start < int(row["control_steps"]) <= end for row in sequence_rows)
            cumulative += events
            coverage_rows.append({
                "rollout_iteration_zero_based": iteration,
                "step_start_inclusive": start + 1,
                "step_end_inclusive": end,
                "active_envs_at_start": sum(int(row["control_steps"]) > start for row in sequence_rows),
                "terminal_events": events,
                "cumulative_terminal_events": cumulative,
                "terminal_coverage_fraction": cumulative / len(sequence_rows),
            })

        sequence_path = os.path.join(out_dir, "sequences.csv")
        trace_path = os.path.join(out_dir, "reward_trace.csv")
        coverage_path = os.path.join(out_dir, "rollout_terminal_coverage.csv")
        write_csv(sequence_path, sequence_rows)
        write_csv(trace_path, trace_rows)
        write_csv(coverage_path, coverage_rows)

        steps = np.asarray([row["control_steps"] for row in sequence_rows], dtype=float)
        direct_fraction = np.asarray([
            row["direct_terminal_credit_path_fraction"] for row in sequence_rows], dtype=float)
        terminal_values = np.asarray([
            row["terminal_reward_sum"] for row in sequence_rows], dtype=float)
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "gate": "Gate E5 frozen-policy PPO credit diagnostic",
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_infos": checkpoint_infos,
            "profiles": FACTORY_PROFILE_IDS,
            "envs_per_profile": args.envs_per_profile,
            "surface_seed_base": args.surface_seed_base,
            "path": env.factory_path_metadata,
            "rollout_steps": args.rollout_steps,
            "gamma": args.gamma,
            "comparison_gamma": args.comparison_gamma,
            "completed": len(sequence_rows),
            "safety_ok": sum(bool(row["safety_ok"]) for row in sequence_rows),
            "quality_ok": sum(bool(row["quality_ok"]) for row in sequence_rows),
            "sensor_fault_steps": int(sum(row["sensor_fault_steps"] for row in sequence_rows)),
            "terminal_events": int(sum(row["terminal_reward_events"] for row in sequence_rows)),
            "control_steps_min": int(steps.min()),
            "control_steps_mean": float(steps.mean()),
            "control_steps_max": int(steps.max()),
            "rollout_iterations_to_first_terminal": int(math.ceil(steps.min() / args.rollout_steps)),
            "rollout_iterations_to_all_terminals": int(math.ceil(steps.max() / args.rollout_steps)),
            "mean_direct_terminal_credit_path_fraction": float(direct_fraction.mean()),
            "terminal_reward_mean": float(terminal_values.mean()),
            "terminal_reward_min": float(terminal_values.min()),
            "terminal_reward_max": float(terminal_values.max()),
            "reward_behavior_changed": False,
            "training_performed": False,
            "ppo_performed": False,
            "checkpoint_created": False,
            "champion_promoted": False,
        }
        metadata_path = os.path.join(out_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        readme_path = os.path.join(out_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as handle:
            handle.write(
                "# Gate E5 PPO credit-assignment diagnostic\n\n"
                f"- Frozen champion, factory profiles only: {len(sequence_rows)}/{len(sequence_rows)} "
                "one-pass executions completed.\n"
                f"- Safety: {metadata['safety_ok']}/{len(sequence_rows)}; quality: "
                f"{metadata['quality_ok']}/{len(sequence_rows)}; sensor faults: "
                f"{metadata['sensor_fault_steps']}.\n"
                f"- Episode steps min/mean/max: {metadata['control_steps_min']} / "
                f"{metadata['control_steps_mean']:.2f} / {metadata['control_steps_max']}.\n"
                f"- With {args.rollout_steps}-step PPO rollouts, first/all terminal events occur "
                f"after {metadata['rollout_iterations_to_first_terminal']} / "
                f"{metadata['rollout_iterations_to_all_terminals']} iterations.\n"
                f"- At most the last {args.rollout_steps} steps directly share a terminal-bearing "
                f"GAE rollout: mean path fraction "
                f"{100.0 * metadata['mean_direct_terminal_credit_path_fraction']:.3f}%.\n"
                f"- Terminal reward min/mean/max: {metadata['terminal_reward_min']:.6f} / "
                f"{metadata['terminal_reward_mean']:.6f} / "
                f"{metadata['terminal_reward_max']:.6f}.\n"
                "- Gate E4 used only 480 steps per environment, below even this diagnostic's "
                "minimum episode length, so it contained zero terminal events by construction.\n"
                "- gamma=0.9995 reduces long-horizon decay but does not remove the 48-step rollout "
                "boundary; terminal credit still reaches only the final rollout directly.\n"
                "- Reward/action/physics formulas were not changed. No training or checkpoint "
                "promotion was performed.\n"
            )
        generated = [sequence_path, trace_path, coverage_path, metadata_path, readme_path]
        checksum_rows = [
            {"sha256": sha256(path), "file": path}
            for path in [checkpoint, os.path.abspath(__file__), *generated]
        ]
        write_csv(os.path.join(out_dir, "checksums.csv"), checksum_rows)
        if sha256(checkpoint) != checkpoint_hash:
            raise RuntimeError("frozen champion changed during Gate E5")
        print(f"[Gate E5] wrote {out_dir}; terminal_events={metadata['terminal_events']}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
