"""RobotPolishEnv extension with a large continuous map and centered ROI path."""
from __future__ import annotations

import json

import numpy as np

from learning.polytwin import config as PC

from .planar_roi_diagnostics import (
    PlanarRoiGeometry,
    centered_raster_lines,
    centered_roi_slices,
    diagnose_roi,
    make_centered_roi_patch,
    surface_view,
)
from .planar_roi_polish_env_cfg import PlanarRoiPolishEnvCfg
from .robot_polish_env import RobotPolishEnv


class PlanarRoiPolishEnv(RobotPolishEnv):
    cfg: PlanarRoiPolishEnvCfg

    def __init__(self, cfg: PlanarRoiPolishEnvCfg, render_mode: str | None = None, **kwargs):
        self.roi_geometry = PlanarRoiGeometry(
            map_size_m=tuple(cfg.patch_size_m),
            roi_size_m=tuple(cfg.evaluation_roi_size_m),
            resolution_m=float(cfg.patch_resolution_m),
            pad_radius_m=PC.PAD_RADIUS_M,
            workpiece_extra_m=0.04,
        )
        self.roi_geometry.validate()
        self._roi_slices = centered_roi_slices(self.roi_geometry)
        super().__init__(cfg, render_mode, **kwargs)
        self._configure_centered_roi_path()

    def _configure_centered_roi_path(self) -> None:
        spacing = self.recipe.step_over_spacing_ratio * PC.PAD_DIAMETER_M
        lines = centered_raster_lines(self.roi_geometry, spacing, direction="x")
        self._lines = [(p0, p1) for p0, p1, _ in lines]
        self._line_len = np.asarray([
            float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            for p0, p1 in self._lines
        ])
        self._path_len = float(self._line_len.sum()) * self.recipe.n_passes
        first_uv = np.asarray(self._pos_at_arc(0.0), dtype=np.float64)
        self._prev_uv[:] = first_uv
        print(
            "[PlanarRoiPolishEnv] map="
            f"{self.roi_geometry.map_size_m}m ROI={self.roi_geometry.roi_size_m}m "
            f"lines={len(self._lines)} path_len={self._path_len:.3f}m "
            f"quality_margin={self.roi_geometry.validate()['quality_footprint_margin_x_m']:.3f}m"
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            ids = self.robot._ALL_INDICES
        else:
            import torch
            ids = torch.as_tensor(env_ids, device=self.device).long()
        for i in ids.cpu().tolist():
            # Parent reset increments episode_count after using its seed.
            seed = self.cfg.surface_seed_base + 97 * i + int(self._episode_count[i]) - 1
            self._surfaces[i] = make_centered_roi_patch(self.roi_geometry, seed)
            if self.cfg.use_terminal_reward:
                self._before_metrics[i] = self._evaluate_quality(i)

    def _evaluate_quality(self, i: int) -> dict:
        roi = surface_view(self._surfaces[i], self._roi_slices)
        q = self._model.evaluate(roi)
        g = self._gloss.evaluate(roi)["summary"]
        return {
            "gu": float(g["gu_mean"]),
            "scratch": float(q["max_residual_scratch_um"]),
            "ra": float(q["ra_um"]),
            "rz": float(q["rz_um"]),
            "cc_min": float(q["clearcoat_min_um"]),
            "temperature_mean_c": float(q["temperature_mean_c"]),
            "temperature_peak_c": float(q["temperature_peak_c"]),
            "thermal_damage_mean": float(q["thermal_damage_mean"]),
            "thermal_damage_peak": float(q["thermal_damage_peak"]),
        }

    def _tile_quality_diagnostic(self, i: int, tiles: tuple[int, int] = (5, 5)) -> dict:
        # Reuse the established GU/tile diagnostic against an ROI view without
        # duplicating its clearcoat counterfactual logic.
        original = self._surfaces[i]
        roi = surface_view(original, self._roi_slices)
        self._surfaces[i] = roi
        try:
            out = super()._tile_quality_diagnostic(i, tiles=tiles)
        finally:
            self._surfaces[i] = original

        diag = diagnose_roi(
            original,
            self.roi_geometry,
            waviness_sigma_m=float(self.cfg.diagnostic_waviness_sigma_m),
            under_over_fraction=float(self.cfg.diagnostic_under_over_fraction),
            tiles=tuple(self.cfg.diagnostic_tiles),
        )
        out.update(diag["scalars"])
        out["roi_geometry_json"] = json.dumps(diag["geometry"], sort_keys=True)
        out["roi_tile_maps_json"] = json.dumps({
            name: np.round(values, 8).tolist()
            for name, values in diag["tile_maps"].items()
        }, sort_keys=True)
        return out

