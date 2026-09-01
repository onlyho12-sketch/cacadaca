"""Sequential cylinder/freeform curriculum for the F10-H residual policy."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone


parser = argparse.ArgumentParser()
parser.add_argument("--parent-checkpoint", required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--iterations-per-stage", type=int, default=250)
parser.add_argument("--envs-per-profile", type=int, default=2)
parser.add_argument("--num-steps-per-env", type=int, default=48)
parser.add_argument("--seed", type=int, default=20260901)
parser.add_argument("--surface-seed-base", type=int, default=51000)
parser.add_argument("--headless", action="store_true")
train_args = parser.parse_args()

# Reuse the already smoke-validated isolated environment implementation.  Its
# module owns AppLauncher argument parsing, so expose only its accepted CLI.
saved_argv = sys.argv
sys.argv = [saved_argv[0], "--parent-checkpoint", train_args.parent_checkpoint,
            "--out-dir", train_args.out_dir]
if train_args.headless:
    sys.argv.append("--headless")
import learning.rl.gate_f10h_residual_ppo_smoke as smoke  # noqa: E402
sys.argv = saved_argv

import numpy as np  # noqa: E402
import torch  # noqa: E402
from importlib import metadata  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


STAGES = (
    ("cylinder", "same_xx"),
    ("freeform", "same_xx"),
    ("cylinder", "cross_xy"),
    ("freeform", "cross_xy"),
)


def make_cfg(kind, direction, stage_index):
    cfg = smoke.GateF7CurvedObservationEnvCfg()
    cfg.factory_profile_ids = smoke.PROFILE_IDS
    cfg.factory_envs_per_profile = train_args.envs_per_profile
    cfg.scene.num_envs = len(smoke.PROFILE_IDS) * train_args.envs_per_profile
    cfg.factory_path_mode = "gate_c"
    cfg.factory_gate_c_direction_mode = direction
    cfg.surface_kind = kind
    cfg.curvature_radius_m = 0.60
    cfg.align_pad_to_surface_normal = True
    cfg.gate_f7_observation_mode = smoke.NORMAL_CURVATURE20
    cfg.observation_space = smoke.observation_dim(smoke.NORMAL_CURVATURE20)
    cfg.surface_seed_base = train_args.surface_seed_base + stage_index * 1000
    cfg.seed = train_args.seed + stage_index
    cfg.enable_pad_physical_contact = True
    cfg.repolish_max_passes = 1
    cfg.episode_length_s = 840.0
    return cfg


def make_agent_cfg(stage_index):
    cfg = smoke.PolishPPORunnerCfg()
    cfg.seed = train_args.seed + stage_index
    cfg.num_steps_per_env = train_args.num_steps_per_env
    cfg.max_iterations = train_args.iterations_per_stage
    cfg.save_interval = train_args.iterations_per_stage
    cfg.experiment_name = f"gate_f10h_stage_{stage_index}"
    cfg.algorithm.learning_rate = 1.0e-4
    cfg.algorithm.clip_param = 0.1
    cfg.algorithm.desired_kl = 0.005
    cfg.algorithm.gamma = 0.9995
    cfg.algorithm.entropy_coef = 0.005
    return handle_deprecated_rsl_rl_cfg(cfg, metadata.version("rsl-rl-lib"))


def main():
    if train_args.iterations_per_stage <= 5:
        raise ValueError("iterations-per-stage must exceed five warmup iterations")
    parent = os.path.abspath(train_args.parent_checkpoint)
    out_dir = os.path.abspath(train_args.out_dir)
    if not os.path.isfile(parent):
        raise FileNotFoundError(parent)
    if os.path.exists(out_dir):
        raise FileExistsError(out_dir)
    os.makedirs(out_dir)
    parent_hash = smoke.sha256(parent)
    total_iterations = train_args.iterations_per_stage * len(STAGES)
    num_envs = len(smoke.PROFILE_IDS) * train_args.envs_per_profile
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F10H_BOUNDED_RESIDUAL_PPO_CURRICULUM",
        "parent_checkpoint": parent, "parent_checkpoint_sha256": parent_hash,
        "stages": [{"surface_kind": kind, "direction_mode": direction}
                   for kind, direction in STAGES],
        "iterations_per_stage": train_args.iterations_per_stage,
        "total_iterations": total_iterations, "num_envs": num_envs,
        "num_steps_per_env": train_args.num_steps_per_env,
        "nominal_samples": total_iterations * num_envs * train_args.num_steps_per_env,
        "observation_mode": smoke.NORMAL_CURVATURE20, "observation_dim": 20,
        "production_training": True, "promotion_allowed": False,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)

    previous_model = None
    stage_results = []
    for stage_index, (kind, direction) in enumerate(STAGES):
        stage_dir = os.path.join(out_dir, f"stage_{stage_index}_{kind}_{direction}")
        os.makedirs(stage_dir)
        env = None
        try:
            smoke.ResidualPPOEnv.parent_checkpoint = parent
            torch.manual_seed(train_args.seed + stage_index)
            env = smoke.ResidualPPOEnv(
                make_cfg(kind, direction, stage_index), render_mode=None)
            env._repolish_mode = True
            agent_cfg = make_agent_cfg(stage_index)
            runner = OnPolicyRunner(
                RslRlVecEnvWrapper(env, clip_actions=1.0), agent_cfg.to_dict(),
                log_dir=stage_dir, device=str(env.device))
            if previous_model is None:
                last_linear = [module for module in runner.alg.actor.modules()
                               if isinstance(module, torch.nn.Linear)][-1]
                torch.nn.init.zeros_(last_linear.weight)
                torch.nn.init.zeros_(last_linear.bias)
                for parameter in runner.alg.actor.parameters():
                    parameter.requires_grad_(False)
                runner.learn(num_learning_iterations=5, init_at_random_ep_len=False)
                runner.current_learning_iteration += 1
                for parameter in runner.alg.actor.parameters():
                    parameter.requires_grad_(True)
                remaining = train_args.iterations_per_stage - 5
                runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=False)
            else:
                runner.load(
                    previous_model,
                    load_cfg={"actor": True, "critic": True, "optimizer": False,
                              "iteration": False, "rnd": False},
                    map_location=str(env.device))
                runner.learn(
                    num_learning_iterations=train_args.iterations_per_stage,
                    init_at_random_ep_len=False)
            checks = {
                "actor_finite": smoke.finite(runner.alg.actor),
                "critic_finite": smoke.finite(runner.alg.critic),
                "cap_contract_failures": env._f10h_cap_contract_failures,
                "supervisor_faults": env._f10h_supervisor_faults,
                "force_overload_seen": env._f10h_force_overload_seen,
                "sensor_faults_current": int(env._sensor_fault.sum().item()),
                "parent_checkpoint_unchanged": smoke.sha256(parent) == parent_hash,
            }
            passed = (checks["actor_finite"] and checks["critic_finite"]
                      and checks["parent_checkpoint_unchanged"]
                      and all(checks[name] == 0 for name in (
                          "cap_contract_failures", "supervisor_faults",
                          "force_overload_seen", "sensor_faults_current")))
            stage_result = {
                "stage_index": stage_index, "surface_kind": kind,
                "direction_mode": direction, "checks": checks,
                "stage_pass": bool(passed),
                "mean_abs_executed_residual_action": (
                    env._f10h_residual_abs_sum / max(1, env._f10h_residual_count)),
                "supervisor_interventions": env._f10h_supervisor_interventions,
            }
            previous_model = os.path.join(stage_dir, "residual_model.pt")
            runner.save(previous_model, infos=stage_result)
            stage_result["model"] = previous_model
            stage_result["model_sha256"] = smoke.sha256(previous_model)
            with open(os.path.join(stage_dir, "stage_result.json"), "w", encoding="utf-8") as fh:
                json.dump(stage_result, fh, indent=2, sort_keys=True)
            stage_results.append(stage_result)
            print(json.dumps(stage_result, indent=2, sort_keys=True), flush=True)
            if not passed:
                raise RuntimeError(f"F10-H training stage {stage_index} failed")
        finally:
            if env is not None:
                env.close()
    result = {
        **config, "training_pass": True, "stage_results": stage_results,
        "final_candidate_model": previous_model,
        "final_candidate_model_sha256": smoke.sha256(previous_model),
        "parent_checkpoint_sha256_after": smoke.sha256(parent),
        "champion_promoted": False, "release_modified": False,
        "evaluation_required": True,
    }
    with open(os.path.join(out_dir, "train_result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        smoke.app.close()
