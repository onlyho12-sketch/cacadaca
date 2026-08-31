"""Pure-Python Gate B tests for planar ROI geometry and metric separation."""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, _REPO_ROOT)

from learning.rl.env.planar_roi_diagnostics import (
    PlanarRoiGeometry,
    centered_raster_lines,
    centered_roi_slices,
    diagnose_roi,
    make_centered_roi_patch,
    surface_view,
)
from learning.polytwin.surface_state import make_flat_patch


def main():
    g = PlanarRoiGeometry()
    contract = g.validate()
    assert contract["quality_footprint_inside"]
    assert contract["physical_footprint_inside"]
    assert abs(contract["quality_footprint_margin_x_m"] - 0.005) < 1e-12
    assert abs(contract["physical_footprint_margin_x_m"] - 0.025) < 1e-12

    sl = centered_roi_slices(g)
    assert (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop) == (30, 130, 30, 130)

    lines = centered_raster_lines(g, 0.184 * 0.110)
    assert len(lines) == 10
    centers = np.asarray([p for p0, p1, _ in lines for p in (p0, p1)])
    assert centers[:, 0].min() - g.pad_radius_m >= -1e-12
    assert centers[:, 1].min() - g.pad_radius_m >= -1e-12
    assert g.map_size_m[0] - (centers[:, 0].max() + g.pad_radius_m) >= -1e-12
    assert g.map_size_m[1] - (centers[:, 1].max() + g.pad_radius_m) >= -1e-12

    seed = 1234
    state = make_centered_roi_patch(g, seed)
    roi = surface_view(state, sl)
    reference = make_flat_patch(g.roi_size_m, g.resolution_m, seed=seed, with_scratches=True)
    support = make_flat_patch(
        g.map_size_m, g.resolution_m, seed=seed + 1_000_003, with_scratches=False)
    assert np.array_equal(roi.initial_scratch_depth_um, reference.initial_scratch_depth_um)
    assert np.array_equal(state.initial_clearcoat_um, support.initial_clearcoat_um)

    # Add a raster-scale low-frequency removal field plus fine ripple and make
    # the current height reflect that removal.  The diagnostic must keep the
    # total/fine and waviness channels distinct.
    nx, ny = roi.shape
    x = np.linspace(-1.0, 1.0, nx)[:, None]
    y = np.linspace(-1.0, 1.0, ny)[None, :]
    low = 1.0 + 0.45 * np.cos(np.pi * x) * np.cos(np.pi * y)
    fine = 0.04 * np.sin(18.0 * np.pi * x) * np.ones((1, ny))
    removal = np.clip(low + fine, 0.0, None)
    roi.cumulative_removal_um[:] = removal
    roi.clearcoat_remaining_um[:] = roi.initial_clearcoat_um - removal
    roi.micro_height_um[:] = roi.initial_micro_height_um - removal

    diag = diagnose_roi(state, g)
    s = diag["scalars"]
    assert s["roi_removal_mean_um"] > 0.0
    assert s["roi_removal_waviness_std_um"] > 0.0
    assert s["roi_removal_fine_std_um"] > 0.0
    assert s["roi_under_fraction"] > 0.0
    assert s["roi_over_fraction"] > 0.0
    assert s["roi_center_edge_ratio"] > 1.0
    assert all(v.shape == (5, 5) for v in diag["tile_maps"].values())
    print("planar ROI diagnostics: 18/18 PASS")


if __name__ == "__main__":
    main()
