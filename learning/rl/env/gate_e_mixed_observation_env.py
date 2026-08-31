"""Gate E mixed legacy/factory environment isolated from frozen Gate B2/D code."""
from __future__ import annotations

import numpy as np

from learning.polytwin.factory_surface_profiles import LEGACY_STRESS, PROFILE_IDS
from learning.polytwin.gloss_proxy import LiteratureGlossProxyModel

from .gate_d_factory_observation_env import GateDFactoryObservationEnv
from .gate_e_mixed_observation_env_cfg import GateEMixedObservationEnvCfg
from .gate_d_observation import OBSERVATION_MODES, observation_dim
from .planar_roi_diagnostics import diagnose_roi, surface_view
from .planar_roi_polish_env import PlanarRoiPolishEnv


class GateEMixedObservationEnv(GateDFactoryObservationEnv):
    """Equal-seed grouped profiles, including the unchanged legacy generator.

    The established Factory class intentionally rejects ``legacy_stress``.
    Gate E needs it only as a regression guard, so this isolated subclass skips
    that factory-only constructor check while reusing its reset/path/diagnostic
    behavior through normal method resolution.
    """

    cfg: GateEMixedObservationEnvCfg

    def __init__(self, cfg: GateEMixedObservationEnvCfg,
                 render_mode: str | None = None, **kwargs):
        profiles = tuple(cfg.factory_profile_ids)
        if not profiles or any(profile not in PROFILE_IDS for profile in profiles):
            raise ValueError(f"Gate E profiles must be drawn from {PROFILE_IDS}")
        if len(set(profiles)) != len(profiles):
            raise ValueError("Gate E profiles must be distinct")
        if int(cfg.factory_envs_per_profile) <= 0:
            raise ValueError("factory_envs_per_profile must be positive")
        expected = len(profiles) * int(cfg.factory_envs_per_profile)
        if int(cfg.scene.num_envs) != expected:
            raise ValueError(
                f"scene.num_envs={cfg.scene.num_envs} but Gate E grouping requires {expected}")
        if cfg.factory_path_mode not in ("gate_b", "gate_c"):
            raise ValueError("factory_path_mode must be gate_b or gate_c")
        if cfg.gate_d_observation_mode not in OBSERVATION_MODES:
            raise ValueError(
                f"gate_d_observation_mode must be one of {OBSERVATION_MODES}")
        expected_obs = observation_dim(str(cfg.gate_d_observation_mode))
        if int(cfg.observation_space) != expected_obs:
            raise ValueError(
                f"observation_space={cfg.observation_space}, expected {expected_obs}")

        self._factory_metadata: dict[int, dict] = {}
        self._factory_gloss = LiteratureGlossProxyModel()
        self._gate_d_inspection_extensions: dict[int, np.ndarray] = {}
        # Deliberately call the planar base constructor.  Dynamic dispatch still
        # reuses Factory reset/path and Gate D observation methods, while avoiding
        # edits to their factory-only validation contract.
        PlanarRoiPolishEnv.__init__(self, cfg, render_mode, **kwargs)
        print(
            "[GateEMixedObservationEnv] "
            f"profiles={profiles} envs/profile={cfg.factory_envs_per_profile} "
            f"path={cfg.factory_path_mode} obs_dim={expected_obs}")

    def factory_diagnostic(self, env_id: int) -> dict:
        """Factory diagnostic with explicit legacy metadata compatibility."""
        state = self._surfaces[env_id]
        diag = diagnose_roi(
            state,
            self.roi_geometry,
            waviness_sigma_m=float(self.cfg.diagnostic_waviness_sigma_m),
            under_over_fraction=float(self.cfg.diagnostic_under_over_fraction),
            tiles=tuple(self.cfg.diagnostic_tiles),
        )
        roi = surface_view(state, self._roi_slices)
        gloss = self._factory_gloss.evaluate(roi)
        metadata = self._factory_metadata[env_id]
        profile_id = str(metadata["profile_id"])
        shallow = metadata.get("shallow_segments", [])
        deep = metadata.get("deep_segments", [])
        # The established legacy generator exposes only its declared count/depth
        # ranges.  Do not reverse-engineer or change it; record -1 for unavailable
        # sampled segment counts and retain its observed map maximum separately.
        is_legacy = profile_id == LEGACY_STRESS
        scalar_metadata = {
            "surface_profile": profile_id,
            "profile_seed": int(metadata["profile_seed"]),
            "design_status": metadata["design_status"],
            "base_ra_target_um": float(metadata["base_ra_target_um"]),
            "shallow_scratch_count": (
                -1 if is_legacy else int(metadata["shallow_scratch_count"])),
            "deep_scratch_count": (
                -1 if is_legacy else int(metadata["deep_scratch_count"])),
            "sampled_shallow_depth_max_um": (
                float(np.max(roi.initial_scratch_depth_um)) if is_legacy else
                max([item["depth_um"] for item in shallow], default=0.0)),
            "sampled_deep_depth_um": (
                0.0 if is_legacy else
                max([item["depth_um"] for item in deep], default=0.0)),
            "legacy_segment_count_unavailable": is_legacy,
            "profile_gu_mean": float(gloss["summary"]["gu_mean"]),
            "profile_gu_p10": float(gloss["summary"]["gu_p10"]),
            "profile_gu_min": float(gloss["summary"]["gu_min"]),
        }
        diag["scalars"].update(scalar_metadata)
        diag["tile_maps"].update({
            "profile_gu": np.asarray(gloss["gu_map"], dtype=float),
            "q_ra": np.asarray(gloss["term_maps"]["q_ra"], dtype=float),
            "q_scratch": np.asarray(gloss["term_maps"]["q_scratch"], dtype=float),
        })
        diag["profile_metadata"] = metadata
        return diag
