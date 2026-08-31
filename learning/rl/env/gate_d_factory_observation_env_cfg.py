"""Isolated Gate D observation-ablation configuration."""
from __future__ import annotations

from isaaclab.utils.configclass import configclass

from .factory_planar_roi_polish_env_cfg import FactoryPlanarRoiPolishEnvCfg
from .gate_d_observation import SPATIAL120, observation_dim


@configclass
class GateDFactoryObservationEnvCfg(FactoryPlanarRoiPolishEnvCfg):
    gate_d_observation_mode: str = SPATIAL120
    gate_d_observation_tiles: tuple = (5, 5)
    observation_space = observation_dim(SPATIAL120)
