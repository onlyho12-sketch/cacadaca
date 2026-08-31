"""Gate F7 mixed-profile curved environment with geometry observations."""
from __future__ import annotations

import numpy as np
import torch

from learning.polytwin.factory_surface_profiles import PROFILE_IDS
from learning.polytwin.gloss_proxy import LiteratureGlossProxyModel
from learning.rl.gate_f7_observation import (
    OBSERVATION_MODES,
    local_geometry_features,
    observation_dim,
)

from .gate_e_mixed_observation_env import GateEMixedObservationEnv
from .gate_f_curved_polish_env import GateFCurvedPolishEnv
from .gate_f7_curved_observation_env_cfg import GateF7CurvedObservationEnvCfg
from .planar_roi_polish_env import PlanarRoiPolishEnv


class GateF7CurvedObservationEnv(GateFCurvedPolishEnv):
    """Isolated Gate F geometry plus legacy/factory mix and F7 observations."""

    cfg: GateF7CurvedObservationEnvCfg

    def __init__(self, cfg: GateF7CurvedObservationEnvCfg,
                 render_mode: str | None = None, **kwargs):
        profiles = tuple(cfg.factory_profile_ids)
        if not profiles or any(profile not in PROFILE_IDS for profile in profiles):
            raise ValueError(f"F7 profiles must be drawn from {PROFILE_IDS}")
        if len(set(profiles)) != len(profiles):
            raise ValueError("F7 profiles must be distinct")
        expected = len(profiles) * int(cfg.factory_envs_per_profile)
        if int(cfg.scene.num_envs) != expected:
            raise ValueError(f"F7 expected {expected} envs, got {cfg.scene.num_envs}")
        mode = str(cfg.gate_f7_observation_mode)
        if mode not in OBSERVATION_MODES:
            raise ValueError(f"F7 mode must be one of {OBSERVATION_MODES}")
        if int(cfg.observation_space) != observation_dim(mode):
            raise ValueError("F7 observation_space does not match mode")

        from learning.rl.gate_f_curved_geometry import SurfaceGeometrySpec
        self._geometry_spec = SurfaceGeometrySpec(
            kind=str(cfg.surface_kind), patch_size_m=tuple(cfg.patch_size_m),
            curvature_radius_m=float(cfg.curvature_radius_m),
            freeform_seed=int(cfg.freeform_seed))
        self._geometry_spec.validate()
        self._factory_metadata = {}
        self._factory_gloss = LiteratureGlossProxyModel()
        PlanarRoiPolishEnv.__init__(self, cfg, render_mode, **kwargs)
        print(f"[GateF7CurvedObservationEnv] mode={mode} obs_dim={observation_dim(mode)}")

    def factory_diagnostic(self, env_id: int) -> dict:
        return GateEMixedObservationEnv.factory_diagnostic(self, env_id)

    def gate_f7_geometry(self) -> torch.Tensor:
        uv = self._pad_uv_actual.detach().cpu().numpy()
        # The actual pad can briefly be outside the map during reset approach.
        # Clamp only the observation query to the declared mesh margin; contact
        # and quality coordinates remain untouched.
        margin = float(self.cfg.curved_mesh_margin_m)
        bounded = np.empty_like(uv)
        bounded[:, 0] = np.clip(uv[:, 0], -margin, self.cfg.patch_size_m[0] + margin)
        bounded[:, 1] = np.clip(uv[:, 1], -margin, self.cfg.patch_size_m[1] + margin)
        values = local_geometry_features(
            self._geometry_spec, bounded[:, 0], bounded[:, 1])
        return torch.as_tensor(values, device=self.device, dtype=torch.float32)

    def _get_observations(self) -> dict:
        base = super()._get_observations()["policy"]
        mode = str(self.cfg.gate_f7_observation_mode)
        if mode == "base14":
            return {"policy": base}
        geometry = self.gate_f7_geometry().to(dtype=base.dtype)
        count = 3 if mode == "normal17" else 6
        policy = torch.cat((base, geometry[:, :count]), dim=1)
        if policy.shape != (self.num_envs, observation_dim(mode)):
            raise RuntimeError(f"unexpected F7 observation shape {tuple(policy.shape)}")
        if not bool(torch.isfinite(policy).all()):
            raise RuntimeError("F7 observation contains NaN/Inf")
        return {"policy": policy}
