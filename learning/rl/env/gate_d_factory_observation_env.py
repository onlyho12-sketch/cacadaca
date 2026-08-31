"""Factory planar environment with cached Gate D inspection observations."""
from __future__ import annotations

import json

import numpy as np
import torch

from .factory_planar_roi_polish_env import FactoryPlanarRoiPolishEnv
from .gate_d_factory_observation_env_cfg import GateDFactoryObservationEnvCfg
from .gate_d_observation import (
    BASE14,
    GLOBAL20,
    OBSERVATION_MODES,
    SPATIAL120,
    GateDObservationTargets,
    encode_inspection_extension,
    observation_dim,
)


class GateDFactoryObservationEnv(FactoryPlanarRoiPolishEnv):
    cfg: GateDFactoryObservationEnvCfg

    def __init__(self, cfg: GateDFactoryObservationEnvCfg,
                 render_mode: str | None = None, **kwargs):
        if cfg.gate_d_observation_mode not in OBSERVATION_MODES:
            raise ValueError(
                f"gate_d_observation_mode must be one of {OBSERVATION_MODES}")
        if tuple(cfg.gate_d_observation_tiles) != (5, 5):
            raise ValueError("Gate D observation tiles must remain 5x5 for this ablation")
        if tuple(cfg.diagnostic_tiles) != tuple(cfg.gate_d_observation_tiles):
            raise ValueError("diagnostic_tiles and gate_d_observation_tiles must match")
        expected = observation_dim(str(cfg.gate_d_observation_mode))
        if int(cfg.observation_space) != expected:
            raise ValueError(
                f"observation_space={cfg.observation_space}, expected {expected} "
                f"for {cfg.gate_d_observation_mode}")
        self._gate_d_inspection_extensions: dict[int, np.ndarray] = {}
        super().__init__(cfg, render_mode, **kwargs)
        print(
            "[GateDFactoryObservationEnv] "
            f"mode={cfg.gate_d_observation_mode} obs_dim={expected} "
            "inspection_cache=initial_then_pass_end")

    def _targets(self) -> GateDObservationTargets:
        return GateDObservationTargets(
            target_gu=float(self.cfg.repolish_target_gu),
            ra_limit_um=float(self.cfg.t_ra_pass_max_um),
            rz_limit_um=float(self.cfg.t_rz_pass_max_um),
            clearcoat_safety_limit_um=float(self.cfg.clearcoat_safety_limit_um),
            max_passes=int(self.cfg.repolish_max_passes),
        )

    def _cache_initial_inspection(self, env_id: int) -> None:
        diagnostic = self.factory_diagnostic(env_id)
        self._gate_d_inspection_extensions[env_id] = encode_inspection_extension(
            diagnostic["scalars"], diagnostic["tile_maps"], 0, self._targets())

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            ids = self.robot._ALL_INDICES
        else:
            ids = torch.as_tensor(env_ids, device=self.device).long()
        for env_id in ids.cpu().tolist():
            self._cache_initial_inspection(env_id)

    def _repolish_decide(self, env_id: int) -> bool:
        terminate = super()._repolish_decide(env_id)
        history = self._repolish_pass_history.get(env_id, [])
        if not history:
            raise RuntimeError("Gate D expected a recorded pass inspection")
        latest = history[-1]
        maps = json.loads(latest["factory_tile_maps_json"])
        self._gate_d_inspection_extensions[env_id] = encode_inspection_extension(
            latest, maps, int(latest["pass"]), self._targets())
        return terminate

    def gate_d_extension(self, env_id: int) -> np.ndarray:
        if env_id not in self._gate_d_inspection_extensions:
            self._cache_initial_inspection(env_id)
        return self._gate_d_inspection_extensions[env_id].copy()

    def _get_observations(self) -> dict:
        base = super()._get_observations()["policy"]
        mode = str(self.cfg.gate_d_observation_mode)
        if mode == BASE14:
            return {"policy": base}
        extensions = np.stack([
            self.gate_d_extension(env_id) for env_id in range(self.num_envs)
        ])
        ext = torch.as_tensor(extensions, device=self.device, dtype=base.dtype)
        if mode == GLOBAL20:
            policy = torch.cat([base, ext[:, :6]], dim=1)
        elif mode == SPATIAL120:
            policy = torch.cat([base, ext], dim=1)
        else:
            raise RuntimeError(f"unknown Gate D observation mode: {mode}")
        if policy.shape != (self.num_envs, observation_dim(mode)):
            raise RuntimeError(f"unexpected Gate D observation shape {policy.shape}")
        if not bool(torch.isfinite(policy).all()):
            raise RuntimeError("Gate D observation contains NaN/Inf")
        return {"policy": policy}
