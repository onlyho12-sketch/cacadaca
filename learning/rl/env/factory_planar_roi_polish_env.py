"""Grouped factory-profile planar environment using only established SurfaceState fields."""
from __future__ import annotations

import json

import numpy as np
import torch

from learning.polytwin.factory_surface_profiles import (
    FACTORY_PROFILE_IDS,
    make_factory_centered_patch,
)
from learning.polytwin.gloss_proxy import LiteratureGlossProxyModel
from learning.rl.gate_c_path_coverage_screen import candidate_lines

from .factory_planar_roi_polish_env_cfg import FactoryPlanarRoiPolishEnvCfg
from .planar_roi_diagnostics import diagnose_roi, surface_view
from .planar_roi_polish_env import PlanarRoiPolishEnv


class FactoryPlanarRoiPolishEnv(PlanarRoiPolishEnv):
    cfg: FactoryPlanarRoiPolishEnvCfg

    def __init__(self, cfg: FactoryPlanarRoiPolishEnvCfg,
                 render_mode: str | None = None, **kwargs):
        profiles = tuple(cfg.factory_profile_ids)
        if not profiles or any(profile not in FACTORY_PROFILE_IDS for profile in profiles):
            raise ValueError(f"factory_profile_ids must be drawn from {FACTORY_PROFILE_IDS}")
        if cfg.factory_envs_per_profile <= 0:
            raise ValueError("factory_envs_per_profile must be positive")
        expected = len(profiles) * int(cfg.factory_envs_per_profile)
        if int(cfg.scene.num_envs) != expected:
            raise ValueError(
                f"scene.num_envs={cfg.scene.num_envs} but grouped profiles require {expected}")
        if cfg.factory_path_mode not in ("gate_b", "gate_c"):
            raise ValueError("factory_path_mode must be gate_b or gate_c")
        self._factory_metadata: dict[int, dict] = {}
        self._factory_gloss = LiteratureGlossProxyModel()
        super().__init__(cfg, render_mode, **kwargs)
        print(
            "[FactoryPlanarRoiPolishEnv] "
            f"profiles={profiles} envs/profile={cfg.factory_envs_per_profile} "
            f"path={cfg.factory_path_mode}")

    def _profile_and_local_id(self, env_id: int) -> tuple[str, int]:
        per = int(self.cfg.factory_envs_per_profile)
        profile_index, local_id = divmod(int(env_id), per)
        return str(self.cfg.factory_profile_ids[profile_index]), local_id

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            ids = self.robot._ALL_INDICES
        else:
            ids = torch.as_tensor(env_ids, device=self.device).long()
        for env_id in ids.cpu().tolist():
            profile_id, local_id = self._profile_and_local_id(env_id)
            # Local id, rather than global env id, gives every profile exactly the
            # same paired seed sequence within a grouped process.
            seed = (
                int(self.cfg.surface_seed_base)
                + 97 * local_id
                + int(self._episode_count[env_id]) - 1
            )
            state, metadata = make_factory_centered_patch(
                profile_id,
                tuple(self.cfg.patch_size_m),
                tuple(self.cfg.evaluation_roi_size_m),
                float(self.cfg.patch_resolution_m),
                seed,
            )
            self._surfaces[env_id] = state
            self._factory_metadata[env_id] = metadata
            if self.cfg.use_terminal_reward:
                self._before_metrics[env_id] = self._evaluate_quality(env_id)

    def _configure_centered_roi_path(self) -> None:
        if self.cfg.factory_path_mode == "gate_b":
            super()._configure_centered_roi_path()
            self.factory_path_metadata = {
                "path_mode": "gate_b",
                "step_over_ratio": float(self.recipe.step_over_spacing_ratio),
                "edge_mode": "established_centered_raster",
                "direction_mode": "same_xx",
                "path_length_m": float(self._path_len),
                "lines": len(self._lines),
            }
            return

        ratio = float(self.cfg.factory_gate_c_step_over_ratio)
        edge_mode = str(self.cfg.factory_gate_c_edge_mode)
        direction_mode = str(self.cfg.factory_gate_c_direction_mode)
        spacing, sweeps = candidate_lines(ratio, edge_mode, direction_mode)
        offset = 0.5 * (
            np.asarray(self.roi_geometry.map_size_m)
            - np.asarray(self.roi_geometry.roi_size_m))
        lines = []
        for _, sweep_lines in sweeps:
            for p0, p1 in sweep_lines:
                lines.append((
                    tuple(np.asarray(p0, dtype=float) + offset),
                    tuple(np.asarray(p1, dtype=float) + offset),
                ))
        self.recipe.step_over_spacing_ratio = ratio
        self._lines = lines
        self._line_len = np.asarray([
            float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            for p0, p1 in lines
        ])
        self._path_len = float(self._line_len.sum())
        first_uv = np.asarray(self._pos_at_arc(0.0), dtype=np.float64)
        self._prev_uv[:] = first_uv
        self.factory_path_metadata = {
            "path_mode": "gate_c",
            "step_over_ratio": ratio,
            "step_over_m": float(spacing),
            "edge_mode": edge_mode,
            "direction_mode": direction_mode,
            "lines_per_sweep": len(sweeps[0][1]),
            "sweeps": len(sweeps),
            "path_length_m": self._path_len,
        }
        print(
            "[FactoryPlanarRoiPolishEnv] Gate C "
            f"so={ratio:.2f} edge={edge_mode} direction={direction_mode} "
            f"path={self._path_len:.3f}m")

    def factory_diagnostic(self, env_id: int) -> dict:
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
        scalar_metadata = {
            "surface_profile": metadata["profile_id"],
            "profile_seed": int(metadata["profile_seed"]),
            "design_status": metadata["design_status"],
            "base_ra_target_um": float(metadata["base_ra_target_um"]),
            "shallow_scratch_count": int(metadata["shallow_scratch_count"]),
            "deep_scratch_count": int(metadata["deep_scratch_count"]),
            "sampled_shallow_depth_max_um": max(
                [item["depth_um"] for item in metadata["shallow_segments"]],
                default=0.0),
            "sampled_deep_depth_um": max(
                [item["depth_um"] for item in metadata["deep_segments"]],
                default=0.0),
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

    def _tile_quality_diagnostic(self, env_id: int,
                                 tiles: tuple[int, int] = (5, 5)) -> dict:
        out = super()._tile_quality_diagnostic(env_id, tiles=tiles)
        factory = self.factory_diagnostic(env_id)
        out.update(factory["scalars"])
        out["factory_metadata_json"] = json.dumps(
            factory["profile_metadata"], sort_keys=True)
        out["factory_tile_maps_json"] = json.dumps({
            name: np.round(values, 8).tolist()
            for name, values in factory["tile_maps"].items()
        }, sort_keys=True)
        return out

