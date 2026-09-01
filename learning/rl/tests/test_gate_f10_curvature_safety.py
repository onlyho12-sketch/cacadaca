"""CPU tests for F10-F curvature safety and bounded residual mapping."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from learning.rl import gate_f10_curvature_safety as cs  # noqa: E402


def test_flat_zero_residual_parity():
    cfg = cs.SafetyConfig()
    force, feed = cs.deterministic_envelope(np.array([0.0]), np.array([False]),
                                            np.array([1.0]), cfg)
    out = cs.apply_bounded_residual(force, feed, np.array([0.0]),
                                    np.zeros((1, 2)), np.array([False]), cfg)
    assert force[0] == cfg.top_flat_force_n
    assert feed[0] == cfg.base_feed_mm_s
    assert out["force_command_n"][0] == force[0]
    assert out["feed_command_mm_s"][0] == feed[0]


def test_risk_derates_top_and_side_continuously():
    cfg = cs.SafetyConfig()
    risk = np.linspace(0.0, 1.0, 20)
    top, feed = cs.deterministic_envelope(risk, np.zeros(20, bool), np.ones(20), cfg)
    side, _ = cs.deterministic_envelope(risk, np.ones(20, bool), np.ones(20), cfg)
    assert np.all(np.diff(top) <= 0.0)
    assert np.all(np.diff(side) <= 0.0)
    assert np.all(np.diff(feed) <= 0.0)
    assert top[-1] == pytest.approx(cfg.top_risk_force_n)
    assert side[-1] == pytest.approx(cfg.side_risk_force_n)


def test_positive_residual_shrinks_to_zero_at_high_risk():
    cfg = cs.SafetyConfig()
    force = np.array([6.0, 5.0]); feed = np.array([10.0, 8.0])
    out = cs.apply_bounded_residual(force, feed, np.array([0.0, 1.0]),
                                    np.ones((2, 2)), np.zeros(2, bool), cfg)
    assert out["force_delta_n"][0] > 0.0
    assert out["force_delta_n"][1] == 0.0
    assert out["feed_delta_mm_s"][1] == 0.0


def test_extreme_actions_stay_inside_hard_envelopes():
    cfg = cs.SafetyConfig()
    force = np.array([5.0, 3.5]); feed = np.array([8.0, 6.35])
    for action in (-100.0, 100.0):
        out = cs.apply_bounded_residual(force, feed, np.full(2, 0.5),
                                        np.full((2, 2), action), np.array([False, True]), cfg)
        assert np.all(out["force_command_n"] >= cfg.entry_force_floor_n)
        assert out["force_command_n"][0] <= cfg.top_flat_force_n
        assert out["force_command_n"][1] <= cfg.side_flat_force_n
        assert np.all(out["feed_command_mm_s"] >= cfg.risk_feed_mm_s)
        assert np.all(out["feed_command_mm_s"] <= cfg.base_feed_mm_s)


def test_slew_resets_at_new_contact_group():
    target = np.array([2.0, 8.0, 8.0])
    out = cs.slew_sequence(target, np.array([0.1, 0.1, 0.1]), 2.0,
                           np.array([True, False, True]))
    assert out[1] == pytest.approx(2.2)
    assert out[2] == 8.0


def test_observation_is_finite_bounded_and_signed():
    cfg = cs.SafetyConfig()
    obs = cs.compose_observation(-20.0, 20.0, 2.0, 50.0, 0.8, 1.5, -1.0,
                                 -0.04, 20.0, -2000.0, cfg)
    assert obs.shape == (10,)
    assert np.isfinite(obs).all()
    assert np.max(np.abs(obs)) <= 1.0
    assert obs[0] == -1.0 and obs[1] == 1.0


def test_geometry_risk_increases_with_each_hazard():
    cfg = cs.SafetyConfig()
    safe = cs.geometry_risk(pad_curvature_ratio=0.0, normal_spread_deg=0.0,
                            fit_confidence=1.0, boundary_risk=0.0, hole_risk=0.0, cfg=cfg)
    risky = cs.geometry_risk(pad_curvature_ratio=0.8, normal_spread_deg=0.0,
                             fit_confidence=1.0, boundary_risk=0.0, hole_risk=0.0, cfg=cfg)
    assert safe == 0.0 and risky == 1.0


def test_disallowed_geometry_emits_zero_commands():
    cfg = cs.SafetyConfig()
    out = cs.apply_bounded_residual(
        np.array([5.0]), np.array([8.0]), np.array([0.8]), np.ones((1, 2)),
        np.array([False]), cfg, command_allowed=np.array([False]))
    assert out["force_command_n"][0] == 0.0
    assert out["feed_command_mm_s"][0] == 0.0


def test_physx_shield_flat_is_exact_parent_parity():
    parent = np.array([[0.25, -0.75], [-1.0, 1.0]], dtype=np.float32)
    out = cs.apply_curvature_physx_action_shield(
        parent, np.zeros((2, 6)), np.zeros((2, 2)), np.zeros(2),
        surface_kind="flat", patch_size_m=(0.32, 0.32),
        baseline_force_n=5.778, baseline_feed_mm_s=12.7,
        force_ratio_limit=0.3, feed_ratio_limit=0.5, control_dt_s=0.05)
    assert np.array_equal(out["actions"], parent)


def test_physx_shield_curved_is_finite_bounded_and_slewed():
    parent = np.ones((1, 2), dtype=np.float32)
    geometry = np.array([[0, 0, 1, 0.8, 0, 0.8]], dtype=np.float32)
    first = cs.apply_curvature_physx_action_shield(
        parent, geometry, np.array([[0.16, 0.16]]), np.array([0.0]),
        surface_kind="cylinder", patch_size_m=(0.32, 0.32),
        baseline_force_n=5.778, baseline_feed_mm_s=12.7,
        force_ratio_limit=0.3, feed_ratio_limit=0.5, control_dt_s=0.05)
    second = cs.apply_curvature_physx_action_shield(
        -parent, geometry, np.array([[0.16, 0.16]]), np.array([0.01]),
        surface_kind="cylinder", patch_size_m=(0.32, 0.32),
        baseline_force_n=5.778, baseline_feed_mm_s=12.7,
        force_ratio_limit=0.3, feed_ratio_limit=0.5, control_dt_s=0.05,
        previous_force_n=first["force_command_n"],
        previous_feed_mm_s=first["feed_command_mm_s"])
    assert np.isfinite(second["actions"]).all()
    assert np.max(np.abs(second["actions"])) <= 1.0
    assert abs(second["force_command_n"][0] - first["force_command_n"][0]) <= 0.1000001
    assert abs(second["feed_command_mm_s"][0] - first["feed_command_mm_s"][0]) <= 1.2500001


def _v2(parent, *, force=0.0, delta=0.0, previous_force=None, previous_feed=None,
        surface_kind="freeform"):
    base = np.zeros((len(parent), 14), dtype=np.float32)
    base[:, 0] = force / 10.0
    base[:, 2] = delta / 5.0
    return cs.apply_curvature_physx_action_shield_v2(
        np.asarray(parent, dtype=np.float32), base, np.zeros((len(parent), 6)),
        np.full((len(parent), 2), 0.16), np.full(len(parent), 0.1),
        surface_kind=surface_kind, patch_size_m=(0.32, 0.32),
        baseline_force_n=5.778, baseline_feed_mm_s=12.7,
        force_ratio_limit=0.3, feed_ratio_limit=0.5, control_dt_s=0.05,
        previous_force_n=previous_force, previous_feed_mm_s=previous_feed)


def test_v2_flat_exact_parity():
    parent = np.array([[0.9, -0.8], [-1.0, 1.0]], dtype=np.float32)
    out = _v2(parent, force=12.0, delta=2.0, previous_force=np.array([8.0, 8.0]),
              previous_feed=np.array([6.35, 6.35]), surface_kind="flat")
    np.testing.assert_array_equal(out["actions"], parent)


def test_v2_downward_parent_request_is_never_delayed():
    parent = np.array([[-0.8, -0.7]], dtype=np.float32)
    out = _v2(parent, previous_force=np.array([7.5]), previous_feed=np.array([18.0]))
    assert out["actions"][0, 0] <= parent[0, 0] + 1e-7
    assert out["actions"][0, 1] <= parent[0, 1] + 1e-7


def test_v2_only_upward_change_is_slew_limited():
    parent = np.ones((1, 2), dtype=np.float32)
    out = _v2(parent, previous_force=np.array([5.0]), previous_feed=np.array([8.0]))
    assert out["force_command_n"][0] <= 5.1000001
    assert out["feed_command_mm_s"][0] <= 9.2500001


def test_v2_predictive_cap_reacts_to_force_and_positive_delta():
    parent = np.ones((1, 2), dtype=np.float32)
    safe = _v2(parent, force=8.0, delta=0.0)
    risky = _v2(parent, force=9.0, delta=1.0)
    assert safe["predictive_force_cap_action"][0] == pytest.approx(1.0)
    assert risky["predictive_force_cap_action"][0] == pytest.approx(0.0, abs=1e-6)
    assert risky["actions"][0, 0] <= 1e-6


def test_v2_rejects_nonfinite_dynamic_observation():
    parent = np.zeros((1, 2), dtype=np.float32)
    base = np.zeros((1, 14), dtype=np.float32); base[0, 2] = np.nan
    with pytest.raises(ValueError):
        cs.apply_curvature_physx_action_shield_v2(
            parent, base, np.zeros((1, 6)), np.zeros((1, 2)), np.zeros(1),
            surface_kind="freeform", patch_size_m=(0.32, 0.32),
            baseline_force_n=5.778, baseline_feed_mm_s=12.7,
            force_ratio_limit=0.3, feed_ratio_limit=0.5, control_dt_s=0.05)
