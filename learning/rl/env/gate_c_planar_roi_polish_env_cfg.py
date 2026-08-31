"""Gate C candidate-path config isolated from Gate B/B2 configs."""
from __future__ import annotations

from isaaclab.utils.configclass import configclass

from .profiled_planar_roi_polish_env_cfg import ProfiledPlanarRoiPolishEnvCfg


@configclass
class GateCPlanarRoiPolishEnvCfg(ProfiledPlanarRoiPolishEnvCfg):
    gate_c_step_over_ratio: float = 0.40
    gate_c_edge_mode: str = "balanced_extend5"
    gate_c_direction_mode: str = "cross_xy"
