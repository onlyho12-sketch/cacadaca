"""Gate F4 PhysX smoke entry point prepared during F1; do not run in F1."""
from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-kind", choices=("flat", "cylinder", "sphere", "freeform"),
                        default="cylinder")
    parser.add_argument("--radius-m", type=float, default=0.60)
    parser.add_argument("--surface-seed", type=int, default=31000)
    parser.add_argument("--freeform-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=700)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    args = _parser().parse_args()
    launcher = AppLauncher(args)
    simulation_app = launcher.app

    import numpy as np
    import torch

    from learning.polytwin.factory_surface_profiles import FACTORY_PREPOLISH
    from learning.rl.env.gate_f_curved_polish_env import GateFCurvedPolishEnv
    from learning.rl.env.gate_f_curved_polish_env_cfg import GateFCurvedPolishEnvCfg

    cfg = GateFCurvedPolishEnvCfg()
    cfg.scene.num_envs = 1
    cfg.factory_profile_ids = (FACTORY_PREPOLISH,)
    cfg.factory_envs_per_profile = 1
    cfg.surface_seed_base = int(args.surface_seed)
    cfg.surface_kind = str(args.surface_kind)
    cfg.curvature_radius_m = float(args.radius_m)
    cfg.freeform_seed = int(args.freeform_seed)
    cfg.align_pad_to_surface_normal = False
    cfg.enable_pad_physical_contact = True

    env = GateFCurvedPolishEnv(cfg, render_mode=None)
    try:
        env.reset()
        action = torch.zeros((1, 2), device=env.device)
        filtered_force = []
        gap = []
        normal_z = []
        sensor_fault_steps = 0
        hard_force_steps = 0
        for _ in range(int(args.steps)):
            env.step(action)
            filtered_force.append(float(env._force_sensor_filt_n[0]))
            gap.append(float(env._pad_gap_m[0]))
            normal_z.append(float(env._pad_surface_normal_w[0, 2]))
            sensor_fault_steps += int(env._sensor_fault.any())
            hard_force_steps += int((env._force_sensor_n > env.cfg.force_hard_limit_n).any())
        state = env._surfaces[0]
        result = {
            "gate": "F4",
            "f1_prepared_only": False,
            "surface_kind": cfg.surface_kind,
            "curvature_radius_m": cfg.curvature_radius_m,
            "steps": int(args.steps),
            "force_filtered_mean_n": float(np.mean(filtered_force)),
            "force_filtered_max_n": float(np.max(filtered_force)),
            "pad_gap_min_m": float(np.min(gap)),
            "pad_gap_max_m": float(np.max(gap)),
            "surface_normal_z_min": float(np.min(normal_z)),
            "sensor_fault_steps": sensor_fault_steps,
            "hard_force_steps": hard_force_steps,
            "no_contact_removal_errors": int(env._no_contact_removal_errors),
            "removal_max_um": float(state.cumulative_removal_um.max()),
            "mesh_vertices": int(getattr(env, "_gate_f_mesh_vertex_count", 0)),
            "mesh_triangles": int(getattr(env, "_gate_f_mesh_triangle_count", 0)),
        }
        result["finite"] = bool(all(
            np.isfinite(value) for key, value in result.items()
            if isinstance(value, (float, int)) and key != "steps"
        ))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["finite"] else 1
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())

