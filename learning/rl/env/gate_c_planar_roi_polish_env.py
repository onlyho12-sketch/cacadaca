"""Profiled planar environment using a selected Gate C candidate path."""
from __future__ import annotations

import numpy as np

from learning.rl.gate_c_path_coverage_screen import candidate_lines

from .gate_c_planar_roi_polish_env_cfg import GateCPlanarRoiPolishEnvCfg
from .profiled_planar_roi_polish_env import ProfiledPlanarRoiPolishEnv


class GateCPlanarRoiPolishEnv(ProfiledPlanarRoiPolishEnv):
    cfg: GateCPlanarRoiPolishEnvCfg

    def _configure_centered_roi_path(self) -> None:
        ratio = float(self.cfg.gate_c_step_over_ratio)
        edge_mode = str(self.cfg.gate_c_edge_mode)
        direction_mode = str(self.cfg.gate_c_direction_mode)
        spacing, sweeps = candidate_lines(ratio, edge_mode, direction_mode)
        offset = 0.5 * (
            np.asarray(self.roi_geometry.map_size_m)
            - np.asarray(self.roi_geometry.roi_size_m))
        lines = []
        for _, sweep_lines in sweeps:
            for p0, p1 in sweep_lines:
                q0 = tuple(np.asarray(p0, dtype=float) + offset)
                q1 = tuple(np.asarray(p1, dtype=float) + offset)
                lines.append((q0, q1))
        self.recipe.step_over_spacing_ratio = ratio
        self._lines = lines
        self._line_len = np.asarray([
            float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            for p0, p1 in self._lines
        ])
        self._path_len = float(self._line_len.sum())
        first_uv = np.asarray(self._pos_at_arc(0.0), dtype=np.float64)
        self._prev_uv[:] = first_uv
        self.gate_c_path_metadata = {
            "step_over_ratio": ratio, "step_over_m": spacing,
            "edge_mode": edge_mode, "direction_mode": direction_mode,
            "lines_per_sweep": len(sweeps[0][1]), "sweeps": len(sweeps),
            "path_length_m": self._path_len,
        }
        print(
            "[GateCPlanarRoiPolishEnv] "
            f"so={ratio:.2f} edge={edge_mode} direction={direction_mode} "
            f"lines/sweep={len(sweeps[0][1])} path={self._path_len:.3f}m")
