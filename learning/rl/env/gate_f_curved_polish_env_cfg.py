"""Isolated configuration for Gate F curved-surface diagnostics."""
from __future__ import annotations

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from learning.polytwin.factory_surface_profiles import FACTORY_PREPOLISH

from .factory_planar_roi_polish_env_cfg import FactoryPlanarRoiPolishEnvCfg


@configclass
class GateFCurvedPolishEnvCfg(FactoryPlanarRoiPolishEnvCfg):
    """Gate F additions without changing any shared planar configuration."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=1.6, replicate_physics=True
    )
    factory_profile_ids: tuple = (FACTORY_PREPOLISH,)
    factory_envs_per_profile: int = 1
    factory_path_mode: str = "gate_c"
    factory_gate_c_direction_mode: str = "same_xx"

    surface_kind: str = "cylinder"
    curvature_radius_m: float = 0.60
    freeform_seed: int = 0
    curved_mesh_grid_size: int = 41
    curved_mesh_margin_m: float = 0.02

    # F1/F4 keeps the pad vertical.  F5 will compare this control against an
    # explicitly validated local-normal-alignment implementation.
    align_pad_to_surface_normal: bool = False
    # F5 remediation: measured steady IK pad-axis bias on curved runs was
    # approximately (+0.42, +0.03) degrees in world tangent X/Y.  Use a
    # conservative inverse command bias; flat parity remains exactly unaltered.
    normal_alignment_command_bias_xy_deg: tuple[float, float] = (-0.35, -0.03)

    # F4 remediation, frozen from the first-run observations before rerun:
    # completed flat/cylinder/sphere showed actual gap 3.15~3.18 mm below the
    # internal clearance command and missed the 4.0 mm p95 gate by at most
    # 0.344 mm.  Add only 0.5 mm upward target feed-forward; force setpoint,
    # controller gains, geometry, and frozen acceptance threshold stay fixed.
    vertical_tracking_compensation_m: float = 0.0005
