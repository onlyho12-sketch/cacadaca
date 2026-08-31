"""Gate E6 equal-sample PPO credit-assignment pilot.

This is an isolated pilot, not a production trainer.  It preserves the frozen
environment/model contracts, starts both arms from the same champion, and
writes into a caller-supplied directory that must not already exist.
"""
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

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--arm", choices=("control", "candidate"), required=True)
parser.add_argument("--num_steps_per_env", type=int, required=True)
parser.add_argument("--gae_lambda", type=float, required=True)
parser.add_argument("--total_iterations", type=int, required=True)
parser.add_argument("--critic_warmup_iterations", type=int, required=True)
parser.add_argument("--envs_per_profile", type=int, default=4)
parser.add_argument("--seed", type=int, default=20260831)
parser.add_argument("--surface_seed_base", type=int, default=18000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from importlib import metadata  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from learning.polytwin.factory_surface_profiles import FACTORY_PROFILE_IDS  # noqa: E402
from learning.rl.env.gate_d_observation import BASE14, observation_dim  # noqa: E402
from learning.rl.env.gate_e_mixed_observation_env import GateEMixedObservationEnv  # noqa: E402
from learning.rl.env.gate_e_mixed_observation_env_cfg import GateEMixedObservationEnvCfg  # noqa: E402
from learning.rl.ppo_cfg import PolishPPORunnerCfg  # noqa: E402


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _drift(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> dict:
    names = [name for name in sorted(set(before) & set(after)) if name.startswith("mlp.")]
    if not names:
        raise RuntimeError("network has no MLP tensors")
    delta2 = base2 = 0.0
    changed = 0
    max_abs = 0.0
    for name in names:
        a = before[name].to(torch.float64)
        b = after[name].to(torch.float64)
        delta = b - a
        delta2 += float(delta.square().sum())
        base2 += float(a.square().sum())
        changed += int(not torch.equal(a, b))
        max_abs = max(max_abs, float(delta.abs().max()))
    return {
        "mlp_tensor_count": len(names),
        "mlp_changed_tensor_count": changed,
        "mlp_l2_delta": delta2 ** 0.5,
        "mlp_relative_l2_delta": (delta2 / max(base2, 1e-30)) ** 0.5,
        "mlp_max_abs_delta": max_abs,
    }


def _finite(module: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in module.state_dict().values())


def main() -> None:
    if args.envs_per_profile <= 0 or args.num_steps_per_env <= 0:
        raise ValueError("environment and rollout counts must be positive")
    if not 0.0 < args.gae_lambda <= 1.0:
        raise ValueError("GAE lambda must be in (0, 1]")
    if not 0 < args.critic_warmup_iterations < args.total_iterations:
        raise ValueError("critic warmup must be between zero and total iterations")
    checkpoint = os.path.abspath(args.checkpoint)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)

    num_envs = len(FACTORY_PROFILE_IDS) * args.envs_per_profile
    actor_iterations = args.total_iterations - args.critic_warmup_iterations
    nominal_samples = num_envs * args.num_steps_per_env * args.total_iterations
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "Gate E6 equal-sample PPO credit pilot",
        "arm": args.arm,
        "pt_design": True,
        "production_training": False,
        "champion_promotion_allowed": False,
        "checkpoint": checkpoint,
        "checkpoint_sha256_before": _sha256(checkpoint),
        "profiles": FACTORY_PROFILE_IDS,
        "num_envs": num_envs,
        "envs_per_profile": args.envs_per_profile,
        "seed": args.seed,
        "surface_seed_base": args.surface_seed_base,
        "observation_mode": BASE14,
        "observation_dim": observation_dim(BASE14),
        "num_steps_per_env": args.num_steps_per_env,
        "gae_lambda": args.gae_lambda,
        "critic_warmup_iterations": args.critic_warmup_iterations,
        "actor_iterations": actor_iterations,
        "total_iterations": args.total_iterations,
        "nominal_samples": nominal_samples,
        "path": {
            "mode": "gate_c", "step_over_ratio": 0.40,
            "edge_mode": "balanced_extend5", "direction_mode": "same_xx",
        },
        "algorithm_overrides": {
            "learning_rate": 1.0e-4, "clip_param": 0.1,
            "desired_kl": 0.005, "gamma": 0.9995,
            "lambda": args.gae_lambda, "entropy_coef": 0.005,
        },
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    env = None
    try:
        cfg = GateEMixedObservationEnvCfg()
        cfg.factory_profile_ids = FACTORY_PROFILE_IDS
        cfg.factory_envs_per_profile = args.envs_per_profile
        cfg.factory_path_mode = "gate_c"
        cfg.factory_gate_c_step_over_ratio = 0.40
        cfg.factory_gate_c_edge_mode = "balanced_extend5"
        cfg.factory_gate_c_direction_mode = "same_xx"
        cfg.gate_d_observation_mode = BASE14
        cfg.observation_space = observation_dim(BASE14)
        cfg.scene.num_envs = num_envs
        cfg.seed = args.seed
        cfg.surface_seed_base = args.surface_seed_base
        cfg.robot_feed_speed_mm_s = 12.7
        cfg.enable_pad_physical_contact = True
        cfg.repolish_max_passes = 1
        cfg.repolish_cooldown_s = 20.0
        cfg.episode_length_s = 840.0
        env = GateEMixedObservationEnv(cfg, render_mode=None)
        env._repolish_mode = True

        agent_cfg = PolishPPORunnerCfg()
        agent_cfg.seed = args.seed
        agent_cfg.num_steps_per_env = args.num_steps_per_env
        agent_cfg.max_iterations = args.total_iterations
        agent_cfg.save_interval = 1
        agent_cfg.experiment_name = f"gate_e6_{args.arm}"
        agent_cfg.algorithm.learning_rate = 1.0e-4
        agent_cfg.algorithm.clip_param = 0.1
        agent_cfg.algorithm.desired_kl = 0.005
        agent_cfg.algorithm.gamma = 0.9995
        agent_cfg.algorithm.lam = args.gae_lambda
        agent_cfg.algorithm.entropy_coef = 0.005
        agent_cfg = handle_deprecated_rsl_rl_cfg(
            agent_cfg, metadata.version("rsl-rl-lib"))

        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        runner = OnPolicyRunner(
            wrapped, agent_cfg.to_dict(), log_dir=out_dir, device=str(env.device))
        parent_info = runner.load(
            checkpoint,
            load_cfg={"actor": True, "critic": True, "optimizer": False,
                      "iteration": True, "rnd": False},
            map_location=str(env.device),
        )
        actor_before = _clone_state(runner.alg.actor)
        critic_before = _clone_state(runner.alg.critic)

        for parameter in runner.alg.actor.parameters():
            parameter.requires_grad_(False)
        print(
            f"[Gate E6 {args.arm}] critic warmup={args.critic_warmup_iterations}; "
            f"rollout={args.num_steps_per_env}; lambda={args.gae_lambda}; envs={num_envs}",
            flush=True)
        runner.learn(
            num_learning_iterations=args.critic_warmup_iterations,
            init_at_random_ep_len=False,
        )
        runner.current_learning_iteration += 1
        for parameter in runner.alg.actor.parameters():
            parameter.requires_grad_(True)
        runner.alg.learning_rate = float(agent_cfg.algorithm.learning_rate)
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = float(agent_cfg.algorithm.learning_rate)
        print(f"[Gate E6 {args.arm}] actor iterations={actor_iterations}", flush=True)
        runner.learn(num_learning_iterations=actor_iterations, init_at_random_ep_len=False)

        expected_last = args.total_iterations - 1
        if runner.current_learning_iteration != expected_last:
            raise RuntimeError(
                f"unexpected iteration provenance {runner.current_learning_iteration} != {expected_last}")
        actor_after = _clone_state(runner.alg.actor)
        critic_after = _clone_state(runner.alg.critic)
        actor_drift = _drift(actor_before, actor_after)
        critic_drift = _drift(critic_before, critic_after)
        checks = {
            "actor_finite": _finite(runner.alg.actor),
            "critic_finite": _finite(runner.alg.critic),
            "actor_mlp_changed": actor_drift["mlp_changed_tensor_count"] > 0,
            "critic_mlp_changed": critic_drift["mlp_changed_tensor_count"] > 0,
            "current_sensor_faults": int(env._sensor_fault.sum().item()),
            "current_fallback_steps": int(env._fallback_steps.sum().item()),
            "current_force_hard_violations": int(env._force_hard_violated.sum().item()),
            "current_thermal_hard_violations": int(env._thermal_hard_violated.sum().item()),
            "current_unstable_hard_violations": int(env._unstable_hard_violated.sum().item()),
            "no_contact_removal_errors": int(env._no_contact_removal_errors),
            "parent_checkpoint_unchanged": _sha256(checkpoint) == config["checkpoint_sha256_before"],
        }
        passed = (
            checks["actor_finite"] and checks["critic_finite"]
            and checks["actor_mlp_changed"] and checks["critic_mlp_changed"]
            and checks["current_sensor_faults"] == 0
            and checks["current_fallback_steps"] == 0
            and checks["current_force_hard_violations"] == 0
            and checks["current_thermal_hard_violations"] == 0
            and checks["current_unstable_hard_violations"] == 0
            and checks["no_contact_removal_errors"] == 0
            and checks["parent_checkpoint_unchanged"]
        )
        final_info = {
            **config,
            "parent_checkpoint_infos": parent_info,
            "actor_drift": actor_drift,
            "critic_drift": critic_drift,
            "integration_checks": checks,
            "completed_iterations": args.total_iterations,
            "last_iteration_index": expected_last,
            "integration_pass": bool(passed),
            "champion_promoted": False,
        }
        final_path = os.path.join(out_dir, "model_final.pt")
        runner.save(final_path, infos=final_info)
        result = {
            **final_info,
            "model_final": final_path,
            "model_final_sha256": _sha256(final_path),
            "checkpoint_sha256_after": _sha256(checkpoint),
        }
        with open(os.path.join(out_dir, "train_result.json"), "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        print(
            f"[Gate E6 {args.arm}] saved={final_path}; pass={passed}; "
            f"actor_relative_l2={actor_drift['mlp_relative_l2_delta']:.8g}", flush=True)
        if not passed:
            raise RuntimeError(f"Gate E6 {args.arm} integration checks failed")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()

