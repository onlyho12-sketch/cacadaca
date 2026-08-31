"""Gate B2 planar ROI environment with isolated surface-profile sidecars."""
from __future__ import annotations

import json

import numpy as np
import torch

from learning.polytwin.mar_optics import evaluate_profile_gloss
from learning.polytwin.surface_profiles import (
    LEGACY_STRESS,
    NEW_CAR_MILD,
    PROFILE_IDS,
    SurfaceProfileSample,
    make_new_car_mild_centered_patch,
)

from .planar_roi_diagnostics import surface_view
from .planar_roi_polish_env import PlanarRoiPolishEnv
from .profiled_planar_roi_diagnostics import (
    diagnose_profile_roi,
    profile_sample_view,
)
from .profiled_planar_roi_polish_env_cfg import ProfiledPlanarRoiPolishEnvCfg


class ProfiledPlanarRoiPolishEnv(PlanarRoiPolishEnv):
    cfg: ProfiledPlanarRoiPolishEnvCfg

    def __init__(self, cfg: ProfiledPlanarRoiPolishEnvCfg,
                 render_mode: str | None = None, **kwargs):
        if cfg.surface_profile not in PROFILE_IDS:
            raise ValueError(f"unknown surface_profile={cfg.surface_profile!r}")
        self._profile_samples: dict[int, SurfaceProfileSample] = {}
        super().__init__(cfg, render_mode, **kwargs)
        print(f"[ProfiledPlanarRoiPolishEnv] profile={cfg.surface_profile}")

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            ids = self.robot._ALL_INDICES
        else:
            ids = torch.as_tensor(env_ids, device=self.device).long()
        for i in ids.cpu().tolist():
            seed = self.cfg.surface_seed_base + 97 * i + int(self._episode_count[i]) - 1
            if self.cfg.surface_profile == NEW_CAR_MILD:
                sample = make_new_car_mild_centered_patch(
                    tuple(self.cfg.patch_size_m), tuple(self.cfg.evaluation_roi_size_m),
                    float(self.cfg.patch_resolution_m), seed)
                self._surfaces[i] = sample.state
            else:
                state = self._surfaces[i]
                sample = SurfaceProfileSample(
                    state=state, profile_id=LEGACY_STRESS,
                    base_micro_height_um=(state.initial_micro_height_um
                                          + state.initial_scratch_depth_um),
                    mar_density=np.zeros(state.shape, dtype=float),
                    mar_severity_initial=np.zeros(state.shape, dtype=float),
                    metadata={
                        "profile_id": LEGACY_STRESS,
                        "profile_seed": seed,
                        "design_status": "ESTABLISHED_GATE_B_GENERATOR_UNCHANGED",
                    },
                )
            sample.validate()
            self._profile_samples[i] = sample
            if self.cfg.use_terminal_reward:
                self._before_metrics[i] = self._evaluate_quality(i)

    def _roi_profile_sample(self, i: int) -> SurfaceProfileSample:
        return profile_sample_view(self._profile_samples[i], self._roi_slices)

    def _evaluate_quality(self, i: int) -> dict:
        # During the base constructor's first reset, the profile sidecar is not
        # installed until after PlanarRoiPolishEnv has created its state.
        if i not in self._profile_samples:
            return super()._evaluate_quality(i)
        roi = self._roi_profile_sample(i)
        quality = self._model.evaluate(roi.state)
        gloss = evaluate_profile_gloss(roi)["summary"]
        return {
            "gu": float(gloss["gu_mean"]),
            "scratch": float(quality["max_residual_scratch_um"]),
            "ra": float(quality["ra_um"]),
            "rz": float(quality["rz_um"]),
            "cc_min": float(quality["clearcoat_min_um"]),
            "temperature_mean_c": float(quality["temperature_mean_c"]),
            "temperature_peak_c": float(quality["temperature_peak_c"]),
            "thermal_damage_mean": float(quality["thermal_damage_mean"]),
            "thermal_damage_peak": float(quality["thermal_damage_peak"]),
        }

    def profile_diagnostic(self, i: int) -> dict:
        return diagnose_profile_roi(
            self._profile_samples[i], self.roi_geometry,
            waviness_sigma_m=float(self.cfg.diagnostic_waviness_sigma_m),
            under_over_fraction=float(self.cfg.diagnostic_under_over_fraction),
            tiles=tuple(self.cfg.diagnostic_tiles),
        )

    def _tile_quality_diagnostic(self, i: int, tiles: tuple[int, int] = (5, 5)) -> dict:
        out = super()._tile_quality_diagnostic(i, tiles=tiles)
        profile = self.profile_diagnostic(i)
        out.update(profile["scalars"])
        out["profile_metadata_json"] = json.dumps(
            profile["profile_metadata"], sort_keys=True)
        out["profile_tile_maps_json"] = json.dumps({
            name: np.round(values, 8).tolist()
            for name, values in profile["tile_maps"].items()
        }, sort_keys=True)
        return out
