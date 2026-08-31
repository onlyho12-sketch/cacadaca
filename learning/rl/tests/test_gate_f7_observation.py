import numpy as np
import pytest

from learning.rl.gate_f_curved_geometry import SurfaceGeometrySpec
from learning.rl.gate_f7_observation import (
    BASE14, NORMAL17, NORMAL_CURVATURE20, compose_observation,
    local_geometry_features, observation_dim, observation_feature_names,
)


@pytest.mark.parametrize("mode,dim", [(BASE14, 14), (NORMAL17, 17), (NORMAL_CURVATURE20, 20)])
def test_schema(mode, dim):
    assert observation_dim(mode) == dim
    assert len(observation_feature_names(mode)) == dim


def test_flat_features_are_exact_contract():
    spec = SurfaceGeometrySpec(kind="flat")
    features = local_geometry_features(spec, np.array([0.0, 0.16]), np.array([0.0, 0.16]))
    np.testing.assert_array_equal(features[:, :3], [[0, 0, 1], [0, 0, 1]])
    np.testing.assert_array_equal(features[:, 3:], 0.0)


def test_curvature_distinguishes_cylinder_and_sphere():
    cylinder = local_geometry_features(
        SurfaceGeometrySpec(kind="cylinder"), 0.16, 0.16)
    sphere = local_geometry_features(
        SurfaceGeometrySpec(kind="sphere"), 0.16, 0.16)
    assert abs(float(cylinder[3])) > 0.1
    assert abs(float(cylinder[5])) < 1e-6
    assert abs(float(sphere[3])) > 0.1 and abs(float(sphere[5])) > 0.1


def test_compose_modes_share_base_prefix():
    base = np.arange(28, dtype=np.float32).reshape(2, 14)
    geometry = np.tile(np.array([0, 0, 1, -0.2, 0, -0.2], dtype=np.float32), (2, 1))
    for mode in (BASE14, NORMAL17, NORMAL_CURVATURE20):
        out = compose_observation(base, geometry, mode)
        np.testing.assert_array_equal(out[:, :14], base)
        assert out.shape == (2, observation_dim(mode))
