"""F10-H isolated curvature-conditioned bounded-residual PPO smoke."""
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
parser.add_argument("--parent-checkpoint", required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--seed", type=int, default=20260901)
parser.add_argument("--surface-seed-base", type=int, default=50000)
parser.add_argument("--num-steps-per-env", type=int, default=48)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from importlib import metadata  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.models.mlp_model import MLPModel  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from learning.polytwin.factory_surface_profiles import PROFILE_IDS  # noqa: E402
from learning.rl.env.gate_f7_curved_observation_env import GateF7CurvedObservationEnv  # noqa: E402
from learning.rl.env.gate_f7_curved_observation_env_cfg import GateF7CurvedObservationEnvCfg  # noqa: E402
from learning.rl.gate_f7_observation import NORMAL_CURVATURE20, observation_dim  # noqa: E402
from learning.rl.gate_f10_bounded_residual import compose_bounded_residual_actions  # noqa: E402
from learning.rl.gate_f10_curvature_safety import (  # noqa: E402
    SafetyConfig, apply_curvature_physx_action_shield_v2)
from learning.rl.gate_f10_substep_supervisor import (  # noqa: E402
    SubstepSafetyState, supervise_substep)
from learning.rl.ppo_cfg import PolishPPORunnerCfg  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clone_mlp(module):
    return {name: value.detach().cpu().clone()
            for name, value in module.state_dict().items() if name.startswith("mlp.")}


def relative_l2(before, after):
    delta2 = base2 = 0.0
    for name in sorted(set(before) & set(after)):
        old = before[name].to(torch.float64)
        new = after[name].to(torch.float64)
        delta2 += float((new - old).square().sum())
        base2 += float(old.square().sum())
    return (delta2 / max(base2, 1e-30)) ** 0.5


def finite(module):
    return all(bool(value.isfinite().all()) for value in module.state_dict().values())


class ResidualPPOEnv(GateF7CurvedObservationEnv):
    parent_checkpoint = ""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._f10h_parent = None
        self._f10h_cached_obs = None
        self._f10h_previous_force = None
        self._f10h_previous_feed = None
        self._f10h_supervisor_state = None
        self._f10h_supervisor_interventions = 0
        self._f10h_supervisor_faults = 0
        self._f10h_cap_contract_failures = 0
        self._f10h_flat_residual_violations = 0
        self._f10h_force_overload_seen = 0
        self._f10h_residual_abs_sum = 0.0
        self._f10h_residual_count = 0
        super().__init__(cfg, render_mode, **kwargs)
        checkpoint = torch.load(
            self.parent_checkpoint, map_location=str(self.device), weights_only=False)
        state = checkpoint["actor_state_dict"]
        if tuple(state["mlp.0.weight"].shape) != (128, 14):
            raise ValueError("frozen parent must be a 14-D actor")
        dummy = TensorDict(
            {"policy": torch.zeros(1, 14, device=self.device)}, batch_size=[1])
        self._f10h_parent = MLPModel(
            dummy, {"actor": ["policy"]}, "actor", 2,
            hidden_dims=[128, 128], activation="elu", obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3,
                              "std_type": "scalar"}).to(self.device)
        self._f10h_parent.load_state_dict(state, strict=True)
        self._f10h_parent.eval()
        for parameter in self._f10h_parent.parameters():
            parameter.requires_grad_(False)
        self._f10h_supervisor_state = SubstepSafetyState.zeros(self.num_envs)

    def _get_observations(self):
        result = super()._get_observations()
        self._f10h_cached_obs = result["policy"].detach().clone()
        return result

    def _pre_physics_step(self, residual_actions):
        if self._f10h_cached_obs is None:
            raise RuntimeError("missing cached observation for residual composition")
        full = self._f10h_cached_obs
        base14 = full[:, :14]
        with torch.no_grad():
            parent = self._f10h_parent(TensorDict(
                {"policy": base14}, batch_size=[self.num_envs])).clamp(-1.0, 1.0)
        parent_np = parent.detach().cpu().numpy()
        shield = apply_curvature_physx_action_shield_v2(
            parent_np, base14.detach().cpu().numpy(),
            self.gate_f7_geometry().detach().cpu().numpy(),
            self._pad_uv_actual.detach().cpu().numpy(), self._arc.detach().cpu().numpy(),
            surface_kind=str(self.cfg.surface_kind), patch_size_m=self.cfg.patch_size_m,
            baseline_force_n=float(self.recipe.target_contact_force_n),
            baseline_feed_mm_s=float(self.recipe.feed_speed_mm_s),
            force_ratio_limit=float(self.cfg.force_ratio_limit),
            feed_ratio_limit=float(self.cfg.feed_ratio_limit),
            control_dt_s=float(self.quality_dt),
            previous_force_n=self._f10h_previous_force,
            previous_feed_mm_s=self._f10h_previous_feed, cfg=SafetyConfig())
        composed = compose_bounded_residual_actions(
            shield["actions"], residual_actions.detach().cpu().numpy(), shield["risk"],
            shield["force_cap_action"], shield["feed_cap_action"],
            surface_kind=str(self.cfg.surface_kind))
        self._f10h_cap_contract_failures += int((~composed["cap_contract"]).sum())
        if str(self.cfg.surface_kind) == "flat":
            self._f10h_flat_residual_violations += int(
                np.any(np.abs(composed["residual_delta"]) > 0.0, axis=1).sum())
        self._f10h_residual_abs_sum += float(np.abs(composed["residual_delta"]).sum())
        self._f10h_residual_count += int(composed["residual_delta"].size)
        combined = composed["actions"]
        baseline_force = float(self.recipe.target_contact_force_n)
        baseline_feed = float(self.recipe.feed_speed_mm_s)
        self._f10h_previous_force = baseline_force * (
            1.0 + combined[:, 0] * float(self.cfg.force_ratio_limit))
        self._f10h_previous_feed = baseline_feed * (
            1.0 + combined[:, 1] * float(self.cfg.feed_ratio_limit))
        super()._pre_physics_step(torch.as_tensor(combined, device=self.device))

    def _apply_action(self):
        if self._f10h_supervisor_state is None:
            return super()._apply_action()
        nominal = self._force_cmd.clone()
        result = supervise_substep(
            nominal.detach().cpu().numpy(),
            self._force_sensor_n.detach().cpu().numpy(),
            self._force_sensor_filt_n.detach().cpu().numpy(),
            self._pad_gap_m.detach().cpu().numpy(),
            surface_kind=str(self.cfg.surface_kind), state=self._f10h_supervisor_state)
        intervention = np.asarray(result["intervention"], dtype=bool)
        self._f10h_supervisor_interventions += int(intervention.sum())
        self._f10h_supervisor_faults += int(np.asarray(result["fault"]).sum())
        self._force_cmd = torch.as_tensor(
            result["force_command_n"], device=self.device, dtype=nominal.dtype)
        retract = torch.as_tensor(
            result["minimum_retract_velocity_m_s"], device=self.device,
            dtype=self.contact.z_vel.dtype)
        mask = torch.as_tensor(intervention, device=self.device)
        self.contact.z_vel = torch.where(
            mask, torch.maximum(self.contact.z_vel, retract), self.contact.z_vel)
        super()._apply_action()
        self._f10h_force_overload_seen += int(self._force_hard_violated.sum().item())

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if self._f10h_supervisor_state is None:
            return
        indices = env_ids.detach().cpu().numpy().astype(int)
        self._f10h_supervisor_state.previous_raw_force_n[indices] = 0.0
        self._f10h_supervisor_state.retract_latched[indices] = False
        self._f10h_supervisor_state.safe_streak[indices] = 0


def main():
    parent = os.path.abspath(args.parent_checkpoint)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isfile(parent):
        raise FileNotFoundError(parent)
    if os.path.exists(out_dir):
        raise FileExistsError(out_dir)
    os.makedirs(out_dir)
    parent_hash = sha256(parent)
    ResidualPPOEnv.parent_checkpoint = parent
    cfg = GateF7CurvedObservationEnvCfg()
    cfg.factory_profile_ids = PROFILE_IDS
    cfg.factory_envs_per_profile = 1
    cfg.scene.num_envs = len(PROFILE_IDS)
    cfg.factory_path_mode = "gate_c"
    cfg.factory_gate_c_direction_mode = "same_xx"
    cfg.surface_kind = "cylinder"
    cfg.curvature_radius_m = 0.60
    cfg.align_pad_to_surface_normal = True
    cfg.gate_f7_observation_mode = NORMAL_CURVATURE20
    cfg.observation_space = observation_dim(NORMAL_CURVATURE20)
    cfg.surface_seed_base = args.surface_seed_base
    cfg.seed = args.seed
    cfg.enable_pad_physical_contact = True
    cfg.repolish_max_passes = 1
    cfg.episode_length_s = 840.0
    run_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F10H_BOUNDED_RESIDUAL_PPO_SMOKE",
        "parent_checkpoint": parent, "parent_checkpoint_sha256": parent_hash,
        "surface_kind": "cylinder", "observation_mode": NORMAL_CURVATURE20,
        "observation_dim": 20, "action_dim": 2, "num_envs": len(PROFILE_IDS),
        "num_steps_per_env": args.num_steps_per_env,
        "critic_warmup_iterations": 1, "actor_iterations": 1,
        "nominal_samples": len(PROFILE_IDS) * args.num_steps_per_env * 2,
        "production_training": False, "promotion_allowed": False,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2, sort_keys=True)
    env = None
    try:
        torch.manual_seed(args.seed)
        env = ResidualPPOEnv(cfg, render_mode=None)
        env._repolish_mode = True
        agent_cfg = PolishPPORunnerCfg()
        agent_cfg.seed = args.seed
        agent_cfg.num_steps_per_env = args.num_steps_per_env
        agent_cfg.max_iterations = 2
        agent_cfg.save_interval = 1
        agent_cfg.experiment_name = "gate_f10h_residual_smoke"
        agent_cfg.algorithm.learning_rate = 1.0e-4
        agent_cfg.algorithm.clip_param = 0.1
        agent_cfg.algorithm.desired_kl = 0.005
        agent_cfg.algorithm.gamma = 0.9995
        agent_cfg.algorithm.entropy_coef = 0.005
        agent_cfg = handle_deprecated_rsl_rl_cfg(
            agent_cfg, metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(
            RslRlVecEnvWrapper(env, clip_actions=1.0), agent_cfg.to_dict(),
            log_dir=out_dir, device=str(env.device))
        # Zero residual mean at initialization; stochastic exploration remains bounded.
        linear = [module for module in runner.alg.actor.modules()
                  if isinstance(module, torch.nn.Linear)][-1]
        torch.nn.init.zeros_(linear.weight)
        torch.nn.init.zeros_(linear.bias)
        actor_before = clone_mlp(runner.alg.actor)
        critic_before = clone_mlp(runner.alg.critic)
        for parameter in runner.alg.actor.parameters():
            parameter.requires_grad_(False)
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        runner.current_learning_iteration += 1
        for parameter in runner.alg.actor.parameters():
            parameter.requires_grad_(True)
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = 1.0e-4
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        checks = {
            "residual_actor_finite": finite(runner.alg.actor),
            "critic_finite": finite(runner.alg.critic),
            "residual_actor_changed": relative_l2(actor_before, clone_mlp(runner.alg.actor)) > 0,
            "critic_changed": relative_l2(critic_before, clone_mlp(runner.alg.critic)) > 0,
            "parent_checkpoint_unchanged": sha256(parent) == parent_hash,
            "cap_contract_failures": env._f10h_cap_contract_failures,
            "supervisor_faults": env._f10h_supervisor_faults,
            "force_overload_seen": env._f10h_force_overload_seen,
            "sensor_faults_current": int(env._sensor_fault.sum().item()),
        }
        passed = (all(checks[name] for name in (
            "residual_actor_finite", "critic_finite", "residual_actor_changed",
            "critic_changed", "parent_checkpoint_unchanged"))
            and all(checks[name] == 0 for name in (
                "cap_contract_failures", "supervisor_faults", "force_overload_seen",
                "sensor_faults_current")))
        result = {
            **run_config, "checks": checks, "smoke_pass": bool(passed),
            "residual_actor_relative_l2": relative_l2(
                actor_before, clone_mlp(runner.alg.actor)),
            "critic_relative_l2": relative_l2(critic_before, clone_mlp(runner.alg.critic)),
            "mean_abs_executed_residual_action": (
                env._f10h_residual_abs_sum / max(1, env._f10h_residual_count)),
            "supervisor_interventions": env._f10h_supervisor_interventions,
            "training_performed": True, "smoke_only": True,
            "champion_promoted": False, "release_modified": False,
        }
        model_path = os.path.join(out_dir, "residual_smoke_model.pt")
        runner.save(model_path, infos=result)
        result["residual_smoke_model"] = model_path
        result["residual_smoke_model_sha256"] = sha256(model_path)
        with open(os.path.join(out_dir, "smoke_result.json"), "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        if not passed:
            raise RuntimeError("F10-H bounded residual PPO smoke failed")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
