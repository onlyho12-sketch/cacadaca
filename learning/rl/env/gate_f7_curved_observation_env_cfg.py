"""Isolated configuration for the Gate F7 curved-observation ablation."""
from __future__ import annotations

from isaaclab.utils.configclass import configclass

from learning.rl.gate_f7_observation import NORMAL_CURVATURE20, observation_dim

from .gate_f_curved_polish_env_cfg import GateFCurvedPolishEnvCfg


@configclass
class GateF7CurvedObservationEnvCfg(GateFCurvedPolishEnvCfg):
    gate_f7_observation_mode: str = NORMAL_CURVATURE20
    observation_space = observation_dim(NORMAL_CURVATURE20)
