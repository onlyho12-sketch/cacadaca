"""Pure regression checks for PT-DESIGN factory pre-polish profiles."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from learning.polytwin.factory_surface_profiles import (  # noqa: E402
    FACTORY_PREPOLISH,
    FACTORY_PREPOLISH_DEEP_DEFECT,
    FACTORY_PREPOLISH_DEEP_STRESS,
    FACTORY_PROFILE_IDS,
    FACTORY_SPECS,
    LEGACY_STRESS,
    make_factory_centered_patch,
)
from learning.polytwin.surface_state import SurfaceState  # noqa: E402
from learning.rl.env.planar_roi_diagnostics import (  # noqa: E402
    PlanarRoiGeometry,
    centered_roi_slices,
    make_centered_roi_patch,
)


def _assert_state_equal(left: SurfaceState, right: SurfaceState) -> None:
    assert vars(left).keys() == vars(right).keys()
    for name in vars(left):
        a, b = getattr(left, name), getattr(right, name)
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b), name
        else:
            assert a == b, name


def main() -> None:
    geometry = PlanarRoiGeometry()
    sl = centered_roi_slices(geometry)

    for seed in range(16):
        direct = make_centered_roi_patch(geometry, seed)
        adapted, metadata = make_factory_centered_patch(
            LEGACY_STRESS, geometry.map_size_m, geometry.roi_size_m,
            geometry.resolution_m, seed)
        _assert_state_equal(direct, adapted)
        assert metadata["design_status"] == "ESTABLISHED_GATE_B_GENERATOR_UNCHANGED"

    for seed in range(32):
        samples = {}
        for profile_id in FACTORY_PROFILE_IDS:
            state, metadata = make_factory_centered_patch(
                profile_id, geometry.map_size_m, geometry.roi_size_m,
                geometry.resolution_m, seed)
            again, again_metadata = make_factory_centered_patch(
                profile_id, geometry.map_size_m, geometry.roi_size_m,
                geometry.resolution_m, seed)
            _assert_state_equal(state, again)
            assert metadata == again_metadata
            assert vars(state).keys() == vars(direct).keys()
            assert not hasattr(state, "mar_density")
            assert not hasattr(state, "mar_severity")
            assert all(
                np.isfinite(value).all()
                for value in vars(state).values() if isinstance(value, np.ndarray))

            spec = FACTORY_SPECS[profile_id]
            assert spec.base_ra_min_um <= metadata["base_ra_target_um"] <= spec.base_ra_max_um
            assert spec.shallow_scratch_count_min <= metadata["shallow_scratch_count"] <= spec.shallow_scratch_count_max
            assert metadata["deep_scratch_count"] == spec.deep_scratch_count
            for segment in metadata["shallow_segments"]:
                assert spec.shallow_depth_min_um <= segment["depth_um"] <= spec.shallow_depth_max_um
                assert spec.scratch_length_min_m <= segment["length_m"] <= spec.scratch_length_max_m
            for segment in metadata["deep_segments"]:
                assert float(spec.deep_depth_min_um) <= segment["depth_um"] <= float(spec.deep_depth_max_um)
            samples[profile_id] = (state, metadata)

        plain = samples[FACTORY_PREPOLISH][0]
        defect = samples[FACTORY_PREPOLISH_DEEP_DEFECT][0]
        stress = samples[FACTORY_PREPOLISH_DEEP_STRESS][0]
        # Same seed means identical base roughness/clearcoat and shallow population.
        for state in (defect, stress):
            assert np.allclose(
                plain.initial_micro_height_um[sl] + plain.initial_scratch_depth_um[sl],
                state.initial_micro_height_um[sl] + state.initial_scratch_depth_um[sl],
                rtol=0.0, atol=1e-15)
            assert np.array_equal(plain.initial_clearcoat_um, state.initial_clearcoat_um)
        assert np.all(defect.initial_scratch_depth_um >= plain.initial_scratch_depth_um)
        assert np.all(stress.initial_scratch_depth_um >= plain.initial_scratch_depth_um)

    print("factory profiles: legacy 16-seed exact + 3 profiles x 32 seeds PASS")


if __name__ == "__main__":
    main()
