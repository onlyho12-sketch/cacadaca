"""Resume an interrupted Gate E7 PPO run from the next saved iteration.

This isolated runner leaves the established Gate E6 pilot and frozen environment
files untouched.  The checkpoint's saved iteration is verified, then training
continues at ``saved_iteration + 1`` so no completed rollout is repeated.
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
parser.add_argument("--expected_saved_iteration", type=int, required=True)
parser.add_argument("--remaining_iterations", type=int, required=True)
parser.add_argument("--cumulative_target_iterations", type=int, required=True)
parser.add_argument("--envs_per_profile", type=int, default=4)
parser.add_argument("--num_steps_per_env", type=int, default=48)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--surface_seed_base", type=int, required=True)
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


def _clone_mlp(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
        if name.startswith("mlp.")
    }


def _relative_l2(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    delta2 = 0.0
    base2 = 0.0
    for name in sorted(set(before) & set(after)):
        old = before[name].to(torch.float64)
        new = after[name].to(torch.float64)
        delta2 += float((new - old).square().sum())
        base2 += float(old.square().sum())
    return float((delta2 / max(base2, 1e-30)) ** 0.5)


def _finite(module: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in module.state_dict().values())


def main() -> None:
    if args.expected_saved_iteration < 0 or args.remaining_iterations <= 0:
        raise ValueError("saved and remaining iteration counts must be valid")
    expected_final_iteration = (
        args.expected_saved_iteration + args.remaining_iterations
    )
    if expected_final_iteration != args.cumulative_target_iterations - 1:
        raise ValueError(
            "resume interval does not end at cumulative_target_iterations - 1: "
            f"{expected_final_iteration} != {args.cumulative_target_iterations - 1}"
        )
    if not 0.0 < args.gae_lambda <= 1.0:
        raise ValueError("GAE lambda must be in (0, 1]")

    checkpoint = os.path.abspath(args.checkpoint)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)

    num_envs = len(FACTORY_PROFILE_IDS) * args.envs_per_profile
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "Gate E7 PPO main training interrupted-run continuation",
        "production_training": True,
        "champion_promotion_allowed": False,
        "checkpoint": checkpoint,
        "checkpoint_sha256_before": _sha256(checkpoint),
        "expected_saved_iteration": args.expected_saved_iteration,
        "resume_at_iteration": args.expected_saved_iteration + 1,
        "remaining_iterations": args.remaining_iterations,
        "cumulative_target_iterations": args.cumulative_target_iterations,
        "segment_nominal_samples": (
            num_envs * args.num_steps_per_env * args.remaining_iterations
        ),
        "cumulative_nominal_samples": (
            num_envs * args.num_steps_per_env * args.cumulative_target_iterations
        ),
        "profiles": FACTORY_PROFILE_IDS,
        "num_envs": num_envs,
        "envs_per_profile": args.envs_per_profile,
        "seed": args.seed,
        "surface_seed_base": args.surface_seed_base,
        "observation_mode": BASE14,
        "observation_dim": observation_dim(BASE14),
        "num_steps_per_env": args.num_steps_per_env,
        "gae_lambda": args.gae_lambda,
        "path": {
            "mode": "gate_c",
            "step_over_ratio": 0.40,
            "edge_mode": "balanced_extend5",
            "direction_mode": "same_xx",
        },
        "algorithm_overrides": {
            "learning_rate": 1.0e-4,
            "clip_param": 0.1,
            "desired_kl": 0.005,
            "gamma": 0.9995,
            "lambda": args.gae_lambda,
            "entropy_coef": 0.005,
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
        agent_cfg.max_iterations = args.cumulative_target_iterations
        agent_cfg.save_interval = 1
        agent_cfg.experiment_name = "gate_e7_resume"
        agent_cfg.algorithm.learning_rate = 1.0e-4
        agent_cfg.algorithm.clip_param = 0.1
        agent_cfg.algorithm.desired_kl = 0.005
        agent_cfg.algorithm.gamma = 0.9995
        agent_cfg.algorithm.lam = args.gae_lambda
        agent_cfg.algorithm.entropy_coef = 0.005
        agent_cfg = handle_deprecated_rsl_rl_cfg(
            agent_cfg, metadata.version("rsl-rl-lib")
        )

        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        runner = OnPolicyRunner(
            wrapped, agent_cfg.to_dict(), log_dir=out_dir, device=str(env.device)
        )
        parent_info = runner.load(
            checkpoint,
            load_cfg={
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": True,
                "rnd": False,
            },
            map_location=str(env.device),
        )
        if runner.current_learning_iteration != args.expected_saved_iteration:
            raise RuntimeError(
                "checkpoint iteration mismatch: "
                f"{runner.current_learning_iteration} != {args.expected_saved_iteration}"
            )

        actor_before = _clone_mlp(runner.alg.actor)
        critic_before = _clone_mlp(runner.alg.critic)
        runner.current_learning_iteration += 1
        runner.alg.learning_rate = float(agent_cfg.algorithm.learning_rate)
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = float(agent_cfg.algorithm.learning_rate)
        print(
            "[Gate E7 resume] "
            f"iterations={runner.current_learning_iteration}..{expected_final_iteration}; "
            f"segment_samples={config['segment_nominal_samples']}",
            flush=True,
        )
        runner.learn(
            num_learning_iterations=args.remaining_iterations,
            init_at_random_ep_len=False,
        )
        if runner.current_learning_iteration != expected_final_iteration:
            raise RuntimeError(
                "unexpected final iteration: "
                f"{runner.current_learning_iteration} != {expected_final_iteration}"
            )

        actor_relative_l2 = _relative_l2(actor_before, _clone_mlp(runner.alg.actor))
        critic_relative_l2 = _relative_l2(critic_before, _clone_mlp(runner.alg.critic))
        checks = {
            "actor_finite": _finite(runner.alg.actor),
            "critic_finite": _finite(runner.alg.critic),
            "actor_changed_in_segment": actor_relative_l2 > 0.0,
            "critic_changed_in_segment": critic_relative_l2 > 0.0,
            "current_sensor_faults": int(env._sensor_fault.sum().item()),
            "current_fallback_steps": int(env._fallback_steps.sum().item()),
            "current_force_hard_violations": int(env._force_hard_violated.sum().item()),
            "current_thermal_hard_violations": int(env._thermal_hard_violated.sum().item()),
            "current_unstable_hard_violations": int(env._unstable_hard_violated.sum().item()),
            "no_contact_removal_errors": int(env._no_contact_removal_errors),
            "parent_checkpoint_unchanged": (
                _sha256(checkpoint) == config["checkpoint_sha256_before"]
            ),
        }
        passed = (
            all(checks[name] for name in (
                "actor_finite",
                "critic_finite",
                "actor_changed_in_segment",
                "critic_changed_in_segment",
                "parent_checkpoint_unchanged",
            ))
            and all(checks[name] == 0 for name in (
                "current_sensor_faults",
                "current_fallback_steps",
                "current_force_hard_violations",
                "current_thermal_hard_violations",
                "current_unstable_hard_violations",
                "no_contact_removal_errors",
            ))
        )
        result = {
            **config,
            "parent_checkpoint_infos": parent_info,
            "completed_final_iteration": expected_final_iteration,
            "actor_relative_l2_delta_in_segment": actor_relative_l2,
            "critic_relative_l2_delta_in_segment": critic_relative_l2,
            "integration_checks": checks,
            "integration_pass": bool(passed),
            "champion_promoted": False,
        }
        final_path = os.path.join(out_dir, "model_final.pt")
        runner.save(final_path, infos=result)
        result.update({
            "model_final": final_path,
            "model_final_sha256": _sha256(final_path),
            "checkpoint_sha256_after": _sha256(checkpoint),
        })
        with open(os.path.join(out_dir, "train_result.json"), "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        print(
            f"[Gate E7 resume] saved={final_path}; pass={passed}; "
            f"actor_relative_l2={actor_relative_l2:.8g}",
            flush=True,
        )
        if not passed:
            raise RuntimeError("Gate E7 resume integration checks failed")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
