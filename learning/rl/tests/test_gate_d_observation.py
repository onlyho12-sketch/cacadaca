"""Isaac-free tests for the Gate D observation ablation."""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, _REPO_ROOT)

from learning.rl.env.gate_d_observation import (
    BASE14,
    GLOBAL20,
    SPATIAL120,
    GateDObservationTargets,
    compose_observation,
    encode_inspection_extension,
    feature_schema_rows,
    observation_dim,
    observation_feature_names,
)


def _diagnostic(scratch_tile=(2, 3), scratch_depth=1.2):
    scalars = {
        "profile_gu_mean": 65.0,
        "roi_total_ra_um": 0.30,
        "roi_total_rz_um": 3.0,
        "roi_clearcoat_min_um": 40.0,
        "roi_center_edge_delta_um": 0.25,
    }
    maps = {
        "scratch_max_um": np.zeros((5, 5)),
        "removal_mean_um": np.full((5, 5), 0.5),
        "under_fraction": np.zeros((5, 5)),
        "over_fraction": np.zeros((5, 5)),
    }
    maps["scratch_max_um"][scratch_tile] = scratch_depth
    maps["under_fraction"][0, 0] = 1.0
    maps["over_fraction"][4, 4] = 0.5
    return scalars, maps


def main():
    assert observation_dim(BASE14) == 14
    assert observation_dim(GLOBAL20) == 20
    assert observation_dim(SPATIAL120) == 120
    assert [len(observation_feature_names(mode)) for mode in
            (BASE14, GLOBAL20, SPATIAL120)] == [14, 20, 120]
    assert len(feature_schema_rows()) == 120

    scalars, maps = _diagnostic()
    targets = GateDObservationTargets(max_passes=4)
    extension = encode_inspection_extension(scalars, maps, 2, targets)
    np.testing.assert_allclose(extension[:6], [0.5, 0.5, 0.5, 0.5, 0.5, 0.25])
    assert extension.shape == (106,) and np.isfinite(extension).all()

    base = np.linspace(-0.7, 0.6, 14, dtype=np.float32)
    base_obs = compose_observation(base, extension, BASE14)
    global_obs = compose_observation(base, extension, GLOBAL20)
    spatial_obs = compose_observation(base, extension, SPATIAL120)
    np.testing.assert_array_equal(global_obs[:14], base_obs)
    np.testing.assert_array_equal(spatial_obs[:20], global_obs)

    # Same global condition but different scratch location: global20 cannot
    # distinguish it, whereas the 5x5 spatial channel can.
    _, maps_shifted = _diagnostic(scratch_tile=(1, 1))
    shifted = encode_inspection_extension(scalars, maps_shifted, 2, targets)
    np.testing.assert_array_equal(extension[:6], shifted[:6])
    assert not np.array_equal(extension[6:31], shifted[6:31])

    bad = {name: values.copy() for name, values in maps.items()}
    bad["removal_mean_um"][0, 0] = np.nan
    try:
        encode_inspection_extension(scalars, bad, 0, targets)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN tile map was accepted")
    print("Gate D observation: schema/prefix/location/finite tests PASS")


if __name__ == "__main__":
    main()
