"""Robot-only config for a continuous planar map with centered evaluation ROI."""
from __future__ import annotations

from isaaclab.utils.configclass import configclass

from .robot_polish_env_cfg import RobotPolishEnvCfg


@configclass
class PlanarRoiPolishEnvCfg(RobotPolishEnvCfg):
    # Full quality state map.  RobotPolishEnv adds 40 mm to this for the physical
    # Workpiece, so the spawned plate is 360 x 360 mm.
    patch_size_m: tuple = (0.32, 0.32)
    evaluation_roi_size_m: tuple = (0.20, 0.20)
    diagnostic_waviness_sigma_m: float = 0.010
    diagnostic_under_over_fraction: float = 0.20
    diagnostic_tiles: tuple = (5, 5)

