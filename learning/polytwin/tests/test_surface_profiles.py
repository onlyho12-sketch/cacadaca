"""Gate B2 regression checks for isolated surface profiles."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from learning.polytwin.mar_optics import (  # noqa: E402
    evaluate_profile_gloss,
    mar_quality_cell_map,
)
from learning.polytwin.roughness_metrics import ra_um  # noqa: E402
from learning.polytwin.surface_profiles import (  # noqa: E402
    LEGACY_STRESS,
    NEW_CAR_MILD,
    NEW_CAR_MILD_SPEC,
    make_new_car_mild_centered_patch,
    make_surface_profile,
)
from learning.polytwin.surface_state import make_flat_patch  # noqa: E402
from learning.rl.env.planar_roi_diagnostics import PlanarRoiGeometry  # noqa: E402
from learning.rl.env.profiled_planar_roi_diagnostics import (  # noqa: E402
    diagnose_profile_roi,
)


def _assert_state_equal(left, right) -> None:
    assert vars(left).keys() == vars(right).keys()
    for name in vars(left):
        a, b = getattr(left, name), getattr(right, name)
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b), name
        else:
            assert a == b, name


def main() -> None:
    # The adapter must not merely look statistically similar; every established
    # SurfaceState array and scalar must remain exactly equal.
    for seed in range(16):
        direct = make_flat_patch((0.20, 0.20), 0.002, seed=seed)
        adapted = make_surface_profile(LEGACY_STRESS, (0.20, 0.20), 0.002, seed).state
        _assert_state_equal(direct, adapted)

    mild = make_surface_profile(NEW_CAR_MILD, (0.20, 0.20), 0.002, 17)
    mild_again = make_surface_profile(NEW_CAR_MILD, (0.20, 0.20), 0.002, 17)
    _assert_state_equal(mild.state, mild_again.state)
    assert np.array_equal(mild.mar_density, mild_again.mar_density)
    assert np.array_equal(mild.mar_severity_initial, mild_again.mar_severity_initial)
    assert mild.metadata == mild_again.metadata

    # Mar is not a height groove.  Initial height contains only base micro texture
    # and the separately retained individual scratch map.
    assert np.allclose(
        mild.state.initial_micro_height_um,
        mild.base_micro_height_um - mild.state.initial_scratch_depth_um,
        rtol=0.0, atol=0.0)
    ra_before = ra_um(mild.state.initial_micro_height_um)
    mar_changed = copy.deepcopy(mild)
    mar_changed.mar_density[:] = 1.0
    mar_changed.mar_severity_initial[:] = 1.0
    assert ra_um(mar_changed.state.initial_micro_height_um) == ra_before

    spec = NEW_CAR_MILD_SPEC
    assert spec.base_micro_ra_min_um <= ra_um(mild.base_micro_height_um) <= spec.base_micro_ra_max_um
    assert 0.0 <= mild.mar_density.min() <= mild.mar_density.max() <= spec.mar_density_max
    assert (0.0 <= mild.mar_severity_initial.min()
            <= mild.mar_severity_initial.max() <= spec.mar_severity_max)
    mild.validate()

    q_initial = mar_quality_cell_map(mild, mild.state.cumulative_removal_um)
    mild.state.cumulative_removal_um[:] = 0.50
    q_after = mar_quality_cell_map(mild, mild.state.cumulative_removal_um)
    assert np.all(q_after >= q_initial)
    assert q_after.mean() > q_initial.mean()

    gloss = evaluate_profile_gloss(mild)
    assert gloss["summary"]["surface_profile"] == NEW_CAR_MILD
    assert gloss["summary"]["mar_optics_used"] is True
    assert np.isfinite(gloss["gu_map"]).all()

    centered = make_new_car_mild_centered_patch((0.32, 0.32), (0.20, 0.20), 0.002, 23)
    assert centered.state.shape == (160, 160)
    assert centered.mar_density.shape == (160, 160)
    # Individual scratches are ROI-local; optical mar and the base field remain continuous.
    outside = centered.state.initial_scratch_depth_um.copy()
    outside[30:130, 30:130] = 0.0
    assert not outside.any()
    assert centered.mar_density[:30].mean() > 0.0

    centered.state.cumulative_removal_um[30:130, 30:130] = 0.35
    profile_diag = diagnose_profile_roi(centered, PlanarRoiGeometry())
    assert profile_diag["scalars"]["surface_profile"] == NEW_CAR_MILD
    assert profile_diag["scalars"]["roi_coverage_fraction"] == 1.0
    assert profile_diag["tile_maps"]["profile_gu"].shape == (5, 5)
    assert profile_diag["tile_maps"]["mar_density_mean"].shape == (5, 5)
    assert all(np.isfinite(values).all()
               for values in profile_diag["tile_maps"].values())

    print("surface profiles: legacy 16-seed exact + new_car_mild 23 checks PASS")


if __name__ == "__main__":
    main()
