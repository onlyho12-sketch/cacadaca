import numpy as np
import pytest

from learning.rl.gate_f10_substep_supervisor import (
    SubstepSafetyConfig,
    SubstepSafetyState,
    supervise_substep,
)


def call(state, raw, *, filtered=None, parent=6.278, gap=-0.00085,
         surface_kind="cylinder"):
    raw = np.atleast_1d(raw).astype(float)
    n = len(raw)
    if filtered is None:
        filtered = raw
    return supervise_substep(
        np.broadcast_to(np.asarray(parent, dtype=float), (n,)), raw,
        np.broadcast_to(np.asarray(filtered, dtype=float), (n,)),
        np.broadcast_to(np.asarray(gap, dtype=float), (n,)),
        surface_kind=surface_kind, state=state)


def test_flat_is_exact_parent_passthrough():
    state = SubstepSafetyState.zeros(2)
    out = call(state, [12.0, 2.0], parent=[6.0, 7.0], surface_kind="flat")
    np.testing.assert_array_equal(out["force_command_n"], [6.0, 7.0])
    assert out["flat_exact_parity"].all()
    assert not out["intervention"].any()


def test_f10g2_event_is_intercepted_before_14n_trip():
    state = SubstepSafetyState.zeros(1)
    samples = [2.061298370361328, 4.214602470397949,
               6.249092102050781, 7.44124698638916, 14.062171936035156]
    trigger_index = None
    for index, raw in enumerate(samples):
        out = call(state, [raw], filtered=[min(raw, 6.4)])
        if out["intervention"][0] and trigger_index is None:
            trigger_index = index
    assert trigger_index == 2
    assert samples[trigger_index] < 14.0


def test_retract_is_latched_until_three_safe_samples():
    state = SubstepSafetyState.zeros(1)
    assert call(state, [6.0])["intervention"][0]
    for _ in range(2):
        out = call(state, [5.0], filtered=[6.0])
        assert out["intervention"][0]
    out = call(state, [5.0], filtered=[6.0])
    assert not out["intervention"][0]


def test_intervention_never_exceeds_parent_and_requests_retract():
    state = SubstepSafetyState.zeros(2)
    out = call(state, [10.0, 13.0], parent=[4.0, 8.0])
    assert np.all(out["force_command_n"] <= np.array([4.0, 8.0]))
    assert np.all(out["force_command_n"] == 0.0)
    assert np.all(out["minimum_retract_velocity_m_s"] == pytest.approx(0.020))


def test_nonfinite_is_fail_closed():
    state = SubstepSafetyState.zeros(1)
    out = call(state, [np.nan])
    assert out["fault"][0]
    assert out["intervention"][0]
    assert out["force_command_n"][0] == 0.0


def test_vector_state_is_independent():
    state = SubstepSafetyState.zeros(2)
    out = call(state, [10.0, 2.0], parent=[6.0, 6.0])
    np.testing.assert_array_equal(out["intervention"], [True, False])


def test_unknown_surface_is_rejected():
    state = SubstepSafetyState.zeros(1)
    with pytest.raises(ValueError):
        call(state, [2.0], surface_kind="concave_unknown")


def test_invalid_config_is_rejected():
    cfg = SubstepSafetyConfig(predictive_guard_n=15.0)
    with pytest.raises(ValueError):
        cfg.validate()


def test_state_shape_must_match():
    state = SubstepSafetyState.zeros(2)
    with pytest.raises(ValueError):
        call(state, [2.0])
