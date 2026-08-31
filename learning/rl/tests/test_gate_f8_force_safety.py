import numpy as np
import pytest

from learning.rl.gate_f8_force_safety import (
    CONTROL, PREDICTIVE_SHIELD, STATIC_CAP, apply_force_action_shield)


def _inputs(force=0.0, delta=0.0):
    actions = np.array([[0.9, -0.25], [-0.7, 0.4]], dtype=np.float32)
    obs = np.zeros((2, 14), dtype=np.float32)
    obs[:, 0] = force / 10.0
    obs[:, 2] = delta / 5.0
    return actions, obs


@pytest.mark.parametrize("mode", [CONTROL, STATIC_CAP, PREDICTIVE_SHIELD])
def test_flat_is_exact_passthrough(mode):
    actions, obs = _inputs(12.0, 2.0)
    out, cap = apply_force_action_shield(actions, obs, mode, curved=False)
    np.testing.assert_array_equal(out, actions)
    np.testing.assert_array_equal(cap, 1.0)


def test_static_cap_changes_only_high_force_action():
    actions, obs = _inputs()
    out, cap = apply_force_action_shield(actions, obs, STATIC_CAP, curved=True)
    np.testing.assert_allclose(out[:, 0], [0.5, -0.7])
    np.testing.assert_array_equal(out[:, 1], actions[:, 1])
    np.testing.assert_allclose(cap, 0.5)


@pytest.mark.parametrize("force,delta,expected", [
    (8.0, 0.0, 0.5),
    (9.0, 0.0, 0.5),
    (10.0, 0.0, 0.5),
    (11.0, 0.0, 0.0),
    (12.0, 0.0, -0.5),
    (9.0, 1.0, 0.0),
])
def test_predictive_cap(force, delta, expected):
    actions, obs = _inputs(force, delta)
    out, cap = apply_force_action_shield(actions, obs, PREDICTIVE_SHIELD, curved=True)
    assert cap[0] == pytest.approx(expected)
    assert out[0, 0] == pytest.approx(min(0.9, expected))
    np.testing.assert_array_equal(out[:, 1], actions[:, 1])


def test_invalid_values_fail_closed():
    actions, obs = _inputs()
    obs[0, 0] = np.nan
    with pytest.raises(ValueError):
        apply_force_action_shield(actions, obs, STATIC_CAP, curved=True)
