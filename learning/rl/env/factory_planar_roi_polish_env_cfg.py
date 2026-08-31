"""Isolated planar configuration for grouped factory pre-polish profiles."""
from __future__ import annotations

from isaaclab.utils.configclass import configclass

from learning.polytwin.factory_surface_profiles import FACTORY_PROFILE_IDS

from .planar_roi_polish_env_cfg import PlanarRoiPolishEnvCfg


@configclass
class FactoryPlanarRoiPolishEnvCfg(PlanarRoiPolishEnvCfg):
    factory_profile_ids: tuple = FACTORY_PROFILE_IDS
    factory_envs_per_profile: int = 4
    factory_path_mode: str = "gate_b"
    factory_gate_c_step_over_ratio: float = 0.40
    factory_gate_c_edge_mode: str = "balanced_extend5"
    factory_gate_c_direction_mode: str = "cross_xy"

