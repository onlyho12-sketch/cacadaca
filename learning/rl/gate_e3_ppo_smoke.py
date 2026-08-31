"""Isolated Gate E3 base14 PPO integration smoke.

This is deliberately not a production training entry point.  It initializes
from the frozen 14-D champion, warms the previously untrained critic, performs
one small actor-update iteration, and writes only to a caller-supplied new
directory.  Surface, reward, action, physics, and path formulas are reused
unchanged from the isolated Gate E environment.
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
parser.add_argument("--envs_per_profile", type=int, default=4)
parser.add_argument("--seed", type=int, default=20260831)
parser.add_argument("--surface_seed_base", type=int, default=12000)
parser.add_argument("--critic_warmup_iterations", type=int, default=2)
parser.add_argument("--actor_iterations", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from importlib import metadata  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from learning.polytwin.factory_surface_profiles import FACTORY_PROFILE_IDS  # noqa: E402
from learning.rl.env.gate_d_observation import BASE14, observation_dim  # noqa: E402
from learning.rl.env.gate_e_mixed_observation_env import GateEMixedObservationEnv  # noqa: E402
from learning.rl.env.gate_e_mixed_observation_env_cfg import (  # noqa: E402
    GateEMixedObservationEnvCfg,
)
from learning.rl.ppo_cfg import PolishPPORunnerCfg  # noqa: E402


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _state_drift(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> dict:
    names = sorted(set(before) & set(after))
    parameter_names = [name for name in names if name.startswith("mlp.")]
    if not parameter_names:
        raise RuntimeError("actor state has no MLP parameters")
    square_delta = 0.0
    square_base = 0.0
    max_abs = 0.0
    finite = True
    changed = 0
    for name in parameter_names:
        a = before[name].to(torch.float64)
        b = after[name].to(torch.float64)
        delta = b - a
        square_delta += float(delta.square().sum())
        square_base += float(a.square().sum())
        max_abs = max(max_abs, float(delta.abs().max()))
        finite &= bool(torch.isfinite(b).all())
        changed += int(not torch.equal(a, b))
    return {
        "mlp_tensor_count": len(parameter_names),
        "mlp_changed_tensor_count": changed,
        "mlp_l2_delta": square_delta ** 0.5,
        "mlp_relative_l2_delta": (square_delta / max(square_base, 1e-30)) ** 0.5,
        "mlp_max_abs_delta": max_abs,
        "mlp_finite": finite,
    }


def _all_finite(module: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in module.state_dict().values())


def main() -> None:
    if args.envs_per_profile <= 0:
        raise ValueError("envs_per_profile must be positive")
    if args.critic_warmup_iterations <= 0 or args.actor_iterations <= 0:
        raise ValueError("warmup and actor iterations must both be positive")
    checkpoint = os.path.abspath(args.checkpoint)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    if os.path.commonpath((checkpoint, out_dir)) == checkpoint:
        raise ValueError("output directory cannot be inside the checkpoint path")
    os.makedirs(out_dir)

    total_iterations = args.critic_warmup_iterations + args.actor_iterations
    num_envs = len(FACTORY_PROFILE_IDS) * args.envs_per_profile
    run_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "Gate E3 base14 PPO integration smoke",
        "pt_design": True,
        "production_training": False,
        "checkpoint": checkpoint,
        "checkpoint_sha256_before": _sha256(checkpoint),
        "profiles": FACTORY_PROFILE_IDS,
        "legacy_training_samples": 0,
        "num_envs": num_envs,
        "envs_per_profile": args.envs_per_profile,
        "seed": args.seed,
        "surface_seed_base": args.surface_seed_base,
        "observation_mode": BASE14,
        "observation_dim": observation_dim(BASE14),
        "path": {
            "mode": "gate_c",
            "step_over_ratio": 0.40,
            "edge_mode": "balanced_extend5",
            "direction_mode": "same_xx",
        },
        "physical_contact": True,
        "num_steps_per_env": 48,
        "critic_warmup_iterations": args.critic_warmup_iterations,
        "actor_iterations": args.actor_iterations,
        "total_iterations": total_iterations,
        "nominal_samples": num_envs * 48 * total_iterations,
        "algorithm_overrides": {
            "learning_rate": 1.0e-4,
            "clip_param": 0.1,
            "desired_kl": 0.005,
            "gamma": 0.9995,
            "entropy_coef": 0.005,
        },
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, sort_keys=True)

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
        agent_cfg.max_iterations = total_iterations
        agent_cfg.save_interval = 1
        agent_cfg.experiment_name = "gate_e3_base14_ppo_smoke"
        agent_cfg.algorithm.learning_rate = 1.0e-4
        agent_cfg.algorithm.clip_param = 0.1
        agent_cfg.algorithm.desired_kl = 0.005
        agent_cfg.algorithm.gamma = 0.9995
        agent_cfg.algorithm.entropy_coef = 0.005
        agent_cfg = handle_deprecated_rsl_rl_cfg(
            agent_cfg, metadata.version("rsl-rl-lib"))

        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        runner = OnPolicyRunner(
            wrapped, agent_cfg.to_dict(), log_dir=out_dir, device=str(env.device))
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
        actor_before = _clone_state(runner.alg.actor)
        critic_before = _clone_state(runner.alg.critic)

        for parameter in runner.alg.actor.parameters():
            parameter.requires_grad_(False)
        print(
            f"[Gate E3 PPO] critic warmup {args.critic_warmup_iterations} iterations; "
            f"actor frozen; envs={num_envs}", flush=True)
        runner.learn(
            num_learning_iterations=args.critic_warmup_iterations,
            init_at_random_ep_len=False,
        )
        # rsl_rl 5.0.1 stores the last zero-based loop index, not the next index.
        # A second learn() call would otherwise repeat that index in logs/checkpoints.
        runner.current_learning_iteration += 1
        for parameter in runner.alg.actor.parameters():
            parameter.requires_grad_(True)
        runner.alg.learning_rate = float(agent_cfg.algorithm.learning_rate)
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = float(agent_cfg.algorithm.learning_rate)
        print(
            f"[Gate E3 PPO] actor update {args.actor_iterations} iteration(s); "
            f"lr={agent_cfg.algorithm.learning_rate}", flush=True)
        runner.learn(
            num_learning_iterations=args.actor_iterations,
            init_at_random_ep_len=False,
        )
        expected_last_iteration = total_iterations - 1
        if runner.current_learning_iteration != expected_last_iteration:
            raise RuntimeError(
                "unexpected rsl_rl iteration provenance: "
                f"{runner.current_learning_iteration} != {expected_last_iteration}")

        actor_after = _clone_state(runner.alg.actor)
        critic_after = _clone_state(runner.alg.critic)
        actor_drift = _state_drift(actor_before, actor_after)
        critic_drift = _state_drift(critic_before, critic_after)
        checks = {
            "parent_actor_loaded_exact": all(
                torch.equal(actor_before[name], value.detach().cpu())
                for name, value in torch.load(
                    checkpoint, map_location="cpu", weights_only=False
                )["actor_state_dict"].items()
            ),
            "actor_finite": _all_finite(runner.alg.actor),
            "critic_finite": _all_finite(runner.alg.critic),
            "actor_mlp_changed": actor_drift["mlp_changed_tensor_count"] > 0,
            "critic_mlp_changed": critic_drift["mlp_changed_tensor_count"] > 0,
            "current_sensor_faults": int(env._sensor_fault.sum().item()),
            "current_fallback_steps": int(env._fallback_steps.sum().item()),
            "current_force_hard_violations": int(env._force_hard_violated.sum().item()),
            "current_thermal_hard_violations": int(env._thermal_hard_violated.sum().item()),
            "current_unstable_hard_violations": int(env._unstable_hard_violated.sum().item()),
            "no_contact_removal_errors": int(env._no_contact_removal_errors),
            "parent_checkpoint_unchanged": (
                _sha256(checkpoint) == run_config["checkpoint_sha256_before"]),
        }
        mandatory = (
            checks["parent_actor_loaded_exact"]
            and checks["actor_finite"]
            and checks["critic_finite"]
            and checks["actor_mlp_changed"]
            and checks["critic_mlp_changed"]
            and checks["current_sensor_faults"] == 0
            and checks["current_fallback_steps"] == 0
            and checks["current_force_hard_violations"] == 0
            and checks["current_thermal_hard_violations"] == 0
            and checks["current_unstable_hard_violations"] == 0
            and checks["no_contact_removal_errors"] == 0
            and checks["parent_checkpoint_unchanged"]
        )
        final_info = {
            **run_config,
            "parent_checkpoint_infos": parent_info,
            "actor_drift": actor_drift,
            "critic_drift": critic_drift,
            "integration_checks": checks,
            "completed_iterations": total_iterations,
            "last_iteration_index": expected_last_iteration,
            "integration_smoke_pass": bool(mandatory),
            "champion_promoted": False,
        }
        final_path = os.path.join(out_dir, "model_final.pt")
        runner.save(final_path, infos=final_info)
        result = {
            **final_info,
            "checkpoint_sha256_after": _sha256(checkpoint),
            "checkpoint_unchanged": _sha256(checkpoint) == run_config["checkpoint_sha256_before"],
            "model_final": final_path,
            "model_final_sha256": _sha256(final_path),
        }
        with open(os.path.join(out_dir, "smoke_result.json"), "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        print(
            f"[Gate E3 PPO] saved {final_path}; integration_pass={mandatory}; "
            f"actor_relative_l2={actor_drift['mlp_relative_l2_delta']:.8g}",
            flush=True,
        )
        if not mandatory:
            raise RuntimeError("Gate E3 PPO integration checks failed")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
