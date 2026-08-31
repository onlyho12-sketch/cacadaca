"""Isolated Gate E teacher-screen configuration.

This configuration only widens the already validated Gate D grouped surface
list to include the frozen legacy stress generator.  It does not alter any
surface, quality, removal, thermal, action, or reward formula.
"""
from __future__ import annotations

from isaaclab.utils.configclass import configclass

from learning.polytwin.factory_surface_profiles import PROFILE_IDS

from .gate_d_factory_observation_env_cfg import GateDFactoryObservationEnvCfg
from .gate_d_observation import SPATIAL120, observation_dim


@configclass
class GateEMixedObservationEnvCfg(GateDFactoryObservationEnvCfg):
    factory_profile_ids: tuple = PROFILE_IDS
    factory_envs_per_profile: int = 4
    gate_d_observation_mode: str = SPATIAL120
    observation_space = observation_dim(SPATIAL120)

