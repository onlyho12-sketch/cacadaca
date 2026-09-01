import numpy as np
import pytest

from learning.rl.gate_f10_bounded_residual import compose_bounded_residual_actions


def compose(parent, residual, *, risk=None, force_cap=None, feed_cap=None,
            kind="cylinder"):
    parent = np.asarray(parent, dtype=float)
    n = len(parent)
    return compose_bounded_residual_actions(
        parent, np.asarray(residual, dtype=float),
        np.zeros(n) if risk is None else np.asarray(risk, dtype=float),
        np.ones(n) if force_cap is None else np.asarray(force_cap, dtype=float),
        np.ones(n) if feed_cap is None else np.asarray(feed_cap, dtype=float),
        surface_kind=kind)


def test_flat_is_exact_parent_and_suppresses_residual():
    parent = np.array([[0.2, -0.3], [1.0, 0.4]])
    out = compose(parent, [[1, 1], [-1, -1]], kind="flat")
    np.testing.assert_array_equal(out["actions"], parent.astype(np.float32))
    assert out["residual_suppressed"].all()


def test_zero_residual_preserves_parent_when_inside_caps():
    parent = np.array([[0.2, -0.3]])
    out = compose(parent, [[0, 0]])
    np.testing.assert_allclose(out["actions"], parent)


def test_residual_physical_bounds_at_zero_risk():
    out = compose([[0, 0], [0, 0]], [[1, 1], [-1, -1]])
    np.testing.assert_allclose(out["actions"], [[0.10, 0.15], [-0.15, -0.20]])


def test_positive_residual_vanishes_at_full_risk():
    out = compose([[0.1, 0.2]], [[1, 1]], risk=[1.0])
    np.testing.assert_allclose(out["actions"], [[0.1, 0.2]])


def test_deterministic_caps_are_authoritative():
    out = compose([[0.2, 0.3]], [[1, 1]], force_cap=[0.22], feed_cap=[0.31])
    np.testing.assert_allclose(out["actions"], [[0.22, 0.31]])
    assert out["cap_contract"].all()


def test_negative_residual_can_drop_below_parent_immediately():
    out = compose([[0.8, 0.8]], [[-1, -1]])
    np.testing.assert_allclose(out["actions"], [[0.65, 0.60]])


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        compose([[0, 0]], [[np.nan, 0]])


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        compose_bounded_residual_actions(
            np.zeros((2, 2)), np.zeros((1, 2)), np.zeros(2),
            np.ones(2), np.ones(2), surface_kind="cylinder")


def test_unknown_surface_rejected():
    with pytest.raises(ValueError):
        compose([[0, 0]], [[0, 0]], kind="concave_unknown")
